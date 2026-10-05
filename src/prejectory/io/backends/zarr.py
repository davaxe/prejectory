from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from prejectory.core.errors import ConfigurationError
from prejectory.core.optional import raise_missing_optional_dependency
from prejectory.core.scene import get_trajectory_schema
from prejectory.io.base import DatasetWriter, split_directory_name
from prejectory.io.encoding import encode_scene_record

try:
    import zarr
    from zarr.codecs import ZstdCodec
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The Zarr scene writer", extra="zarr")

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.config.models import OutputConfig, ZarrOutputConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.core.scene import Scene, TrajectorySchema
    from prejectory.io.records import PredictionBounds, SceneRecord

ZARR_FORMAT_ID = "prejectory.scene.zarr"
ZARR_FORMAT_VERSION = 2

# Sentinel values are only used in scene/index. All index values are int64.
_NONE_I32 = np.int64(-1)
_NONE_I64 = np.int64(np.iinfo(np.int64).min)

# scene/index columns
_SCENE_NUMBER = 0
_DATASET_ID = 1
_EGO_AGENT_ID = 2
_PREDICTION_ORIGIN = 3
_PREDICTION_END = 4
_AGENT_START = 5
_AGENT_END = 6
_MAP_NODE_START = 7
_MAP_NODE_END = 8
_MAP_EDGE_START = 9
_MAP_EDGE_END = 10
_SCENE_INDEX_WIDTH = 11

_SCENE_INDEX_COLUMNS = (
    "scene_number",
    "dataset_id",
    "ego_agent_id",
    "prediction_origin",
    "prediction_end",
    "agent_start",
    "agent_end",
    "map_node_start",
    "map_node_end",
    "map_edge_start",
    "map_edge_end",
)

# Buffering greatly reduces repeated resize/recompression of tail chunks while
# bounding peak memory. The scene-count limit is additionally capped by
# config.scene_chunk in _ZarrShard.create().
_MAX_BUFFERED_SCENES = 256
_MAX_BUFFERED_BYTES = 64 * 1024 * 1024

ZarrArray: TypeAlias = zarr.Array[ArrayV2Metadata] | zarr.Array[ArrayV3Metadata]
Compressors: TypeAlias = tuple[ZstdCodec, ...] | None


class ZarrDatasetWriter(DatasetWriter):
    """Write canonical scene records to worker-local Zarr v3 shards."""

    def __init__(
        self,
        output_dir: Path,
        identifier: int,
        *,
        config: OutputConfig,
        prediction_bounds: PredictionBounds | None = None,
        splits: Iterable[DatasetSplit] | None = None,
    ) -> None:
        self._base_output_dir: Path = output_dir
        self._identifier: str = f"{identifier:05d}"
        self._config: OutputConfig = config
        self._trajectory_schema: TrajectorySchema = get_trajectory_schema(config.trajectory_schema)
        self._prediction_bounds: PredictionBounds | None = prediction_bounds
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._shards: dict[DatasetSplit | None, _ZarrShard] = {}

    @override
    def write(self, scene: Scene) -> None:
        """Encode and buffer one scene in its split-specific worker shard."""
        split = scene.split_assignment
        allowed = self._splits if self._splits is not None else (None,)
        if split not in allowed:
            msg = (
                f"Scene {scene.scene_number} belongs to split {split}, "
                "but no Zarr writer is configured for this split."
            )
            raise ConfigurationError(msg)

        record = encode_scene_record(
            scene,
            dtype=np.float32 if self._config.precision == "float32" else np.float64,
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._trajectory_schema,
            prediction_bounds=self._prediction_bounds,
        )

        shard = self._shards.get(split)
        if shard is None:
            shard = _ZarrShard.create(
                self._shard_path(split),
                record=record,
                config=self._config.zarr,
            )
            self._shards[split] = shard
        shard.append(record)

    def _shard_path(self, split: DatasetSplit | None) -> Path:
        name = f"part-{self._identifier}.zarr"
        return self._base_output_dir / split_directory_name(split) / name

    @override
    def flush_local(self) -> None:
        """Persist buffered records without closing the worker's shards."""
        for shard in self._shards.values():
            shard.flush()

    @override
    def finish_local(self) -> None:
        """Flush remaining worker-local records and release shard handles."""
        self.flush_local()
        self._shards.clear()

    @staticmethod
    def finish_dataset(output_dir: Path, splits: Iterable[DatasetSplit] | None) -> None:
        """Consolidate shard metadata after all workers have exited."""
        for split in splits or (None,):
            split_dir = output_dir / split_directory_name(split)
            for path in split_dir.glob("part-*.zarr"):
                _consolidate_shard(path)


@dataclass(slots=True)
class _ZarrShard:
    path: Path
    root: zarr.Group
    arrays: dict[str, ZarrArray]
    horizon_frames: int
    feature_dim: int
    flush_scene_limit: int
    record_count: int = 0
    agent_count: int = 0
    map_node_count: int = 0
    map_edge_count: int = 0
    _buffer: list[SceneRecord] = field(default_factory=list)
    _buffered_bytes: int = 0

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        record: SceneRecord,
        config: ZarrOutputConfig,
    ) -> _ZarrShard:
        path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(path, mode="w", zarr_format=3)

        horizon = record.horizon_frames
        feature_dim = int(record.features.shape[2])
        root.attrs.update({
            "format": ZARR_FORMAT_ID,
            "format_version": ZARR_FORMAT_VERSION,
            "horizon_frames": horizon,
            "feature_dim": feature_dim,
            "precision": str(record.features.dtype),
            "record_count": 0,
            "agent_count": 0,
            "map_node_count": 0,
            "map_edge_count": 0,
            "scene_index_columns": list(_SCENE_INDEX_COLUMNS),
            "map_edges_rows": ["source", "target", "type"],
        })

        compressors: Compressors = (
            None
            if config.compression_level is None
            else (ZstdCodec(level=config.compression_level),)
        )

        arrays: dict[str, ZarrArray] = {
            "scene/index": _empty_array(
                root,
                "scene/index",
                np.int64,
                (0, _SCENE_INDEX_WIDTH),
                (config.scene_chunk, _SCENE_INDEX_WIDTH),
                compressors,
            ),
            "scene/position_offset": _empty_array(
                root,
                "scene/position_offset",
                np.float64,
                (0, 2),
                (config.scene_chunk, 2),
                compressors,
            ),
            "agent/ids": _empty_array(
                root,
                "agent/ids",
                np.int64,
                (0,),
                (config.agent_chunk,),
                compressors,
            ),
            "agent/types": _empty_array(
                root,
                "agent/types",
                np.int32,
                (0,),
                (config.agent_chunk,),
                compressors,
            ),
            "agent/screened_mask": _empty_array(
                root,
                "agent/screened_mask",
                np.bool_,
                (0,),
                (config.agent_chunk,),
                compressors,
            ),
            "agent/features": _empty_array(
                root,
                "agent/features",
                record.features.dtype,
                (0, horizon, feature_dim),
                (config.agent_chunk, horizon, feature_dim),
                compressors,
            ),
            "agent/valid_mask": _empty_array(
                root,
                "agent/valid_mask",
                np.bool_,
                (0, horizon),
                (config.agent_chunk, horizon),
                compressors,
            ),
            "map/node_positions": _empty_array(
                root,
                "map/node_positions",
                record.map_node_positions.dtype,
                (0, 2),
                (config.map_node_chunk, 2),
                compressors,
            ),
            "map/node_types": _empty_array(
                root,
                "map/node_types",
                np.int32,
                (0,),
                (config.map_node_chunk,),
                compressors,
            ),
            # Rows are [source, target, edge_type]. This gives the reader one
            # contiguous edge read and avoids a per-scene transpose/copy.
            "map/edges": _empty_array(
                root,
                "map/edges",
                np.int32,
                (3, 0),
                (3, config.map_edge_chunk),
                compressors,
            ),
        }

        return cls(
            path=path,
            root=root,
            arrays=arrays,
            horizon_frames=horizon,
            feature_dim=feature_dim,
            flush_scene_limit=max(1, min(config.scene_chunk, _MAX_BUFFERED_SCENES)),
        )

    def append(self, record: SceneRecord) -> None:
        """Buffer a record and flush batches to the shard when appropriate."""
        self._validate_record(record)
        self._buffer.append(record)
        self._buffered_bytes += _record_nbytes(record)

        if (
            len(self._buffer) >= self.flush_scene_limit
            or self._buffered_bytes >= _MAX_BUFFERED_BYTES
        ):
            self.flush()

    def _validate_record(self, record: SceneRecord) -> None:
        if (
            record.horizon_frames != self.horizon_frames
            or record.features.shape[2] != self.feature_dim
        ):
            msg = "All records in a Zarr shard must share horizon and feature dimensions."
            raise ValueError(msg)

        num_agents = int(record.features.shape[0])
        if (
            record.agent_ids.shape[0] != num_agents
            or record.agent_types.shape[0] != num_agents
            or record.screened_agent_mask.shape[0] != num_agents
            or record.valid_mask.shape[0] != num_agents
        ):
            msg = "Agent arrays in a SceneRecord have inconsistent first dimensions."
            raise ValueError(msg)

        num_nodes = int(record.map_node_positions.shape[0])
        if record.map_node_types.shape[0] != num_nodes:
            msg = "Map node arrays in a SceneRecord have inconsistent first dimensions."
            raise ValueError(msg)

        if record.map_edge_indices.ndim != 2 or record.map_edge_indices.shape[0] != 2:
            msg = "SceneRecord.map_edge_indices must have shape (2, E)."
            raise ValueError(msg)
        if record.map_edge_types.shape[0] != record.map_edge_indices.shape[1]:
            msg = "Map edge arrays in a SceneRecord have inconsistent edge dimensions."
            raise ValueError(msg)

    def flush(self) -> None:
        if not self._buffer:
            return

        records = tuple(self._buffer)
        scene_index = self._build_scene_index(records)

        # Payload is written first. scene/index is written last and therefore
        # acts as the commit marker for the whole batch.
        self._append("agent/ids", _concat(records, "agent_ids"))
        self._append("agent/types", _concat(records, "agent_types"))
        self._append("agent/screened_mask", _concat(records, "screened_agent_mask"))
        self._append("agent/features", _concat(records, "features"))
        self._append("agent/valid_mask", _concat(records, "valid_mask"))

        self._append("map/node_positions", _concat(records, "map_node_positions"))
        self._append("map/node_types", _concat(records, "map_node_types"))
        self._append("map/edges", _edge_batch(records), axis=1)

        position_offsets = np.stack(
            [np.asarray(record.position_offset, dtype=np.float64) for record in records],
            axis=0,
        )
        self._append("scene/position_offset", position_offsets)
        self._append("scene/index", scene_index)

        self.record_count += len(records)
        self.agent_count = int(scene_index[-1, _AGENT_END])
        self.map_node_count = int(scene_index[-1, _MAP_NODE_END])
        self.map_edge_count = int(scene_index[-1, _MAP_EDGE_END])
        self.root.attrs.update({
            "record_count": self.record_count,
            "agent_count": self.agent_count,
            "map_node_count": self.map_node_count,
            "map_edge_count": self.map_edge_count,
        })

        self._buffer.clear()
        self._buffered_bytes = 0

    def _build_scene_index(self, records: tuple[SceneRecord, ...]) -> npt.NDArray[np.int64]:
        index = np.empty((len(records), _SCENE_INDEX_WIDTH), dtype=np.int64)

        agent_cursor = self.agent_count
        node_cursor = self.map_node_count
        edge_cursor = self.map_edge_count

        for row, record in zip(index, records, strict=True):
            num_agents = int(record.features.shape[0])
            num_nodes = int(record.map_node_positions.shape[0])
            num_edges = int(record.map_edge_indices.shape[1])

            agent_end = agent_cursor + num_agents
            node_end = node_cursor + num_nodes
            edge_end = edge_cursor + num_edges

            row[_SCENE_NUMBER] = record.scene_number
            row[_DATASET_ID] = _NONE_I32 if record.dataset_id is None else record.dataset_id
            row[_EGO_AGENT_ID] = _NONE_I64 if record.ego_agent_id is None else record.ego_agent_id
            row[_PREDICTION_ORIGIN] = (
                _NONE_I32 if record.prediction_origin is None else record.prediction_origin
            )
            row[_PREDICTION_END] = (
                _NONE_I32 if record.prediction_end is None else record.prediction_end
            )
            row[_AGENT_START] = agent_cursor
            row[_AGENT_END] = agent_end
            row[_MAP_NODE_START] = node_cursor
            row[_MAP_NODE_END] = node_end
            row[_MAP_EDGE_START] = edge_cursor
            row[_MAP_EDGE_END] = edge_end

            agent_cursor = agent_end
            node_cursor = node_end
            edge_cursor = edge_end

        return index

    def _append(self, name: str, values: npt.ArrayLike, *, axis: int = 0) -> None:
        array = self.arrays[name]
        values_array = np.asarray(values)
        if values_array.shape[axis] == 0:
            return
        _ = array.append(values_array, axis=axis)


def _consolidate_shard(path: Path) -> None:
    # The reader also supports shards without consolidated metadata.
    consolidate_metadata = getattr(zarr, "consolidate_metadata", None)
    if consolidate_metadata is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                _ = consolidate_metadata(path, zarr_format=3)
        except (NotImplementedError, TypeError):
            pass


def _empty_array(
    root: zarr.Group,
    name: str,
    dtype: npt.DTypeLike,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    compressors: Compressors,
) -> ZarrArray:
    return root.create_array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressors=compressors,
    )


def _concat(records: tuple[SceneRecord, ...], attribute: str) -> npt.NDArray[np.generic]:
    arrays = [np.asarray(getattr(record, attribute)) for record in records]
    return np.concatenate(arrays, axis=0)


def _edge_batch(records: tuple[SceneRecord, ...]) -> npt.NDArray[np.int32]:
    num_edges = sum(int(record.map_edge_indices.shape[1]) for record in records)
    edges = np.empty((3, num_edges), dtype=np.int32)

    cursor = 0
    for record in records:
        count = int(record.map_edge_indices.shape[1])
        end = cursor + count
        edges[:2, cursor:end] = record.map_edge_indices
        edges[2, cursor:end] = record.map_edge_types
        cursor = end

    return edges


def _record_nbytes(record: SceneRecord) -> int:
    return sum(
        np.asarray(array).nbytes
        for array in (
            record.position_offset,
            record.agent_ids,
            record.agent_types,
            record.screened_agent_mask,
            record.features,
            record.valid_mask,
            record.map_node_positions,
            record.map_node_types,
            record.map_edge_indices,
            record.map_edge_types,
        )
    )
