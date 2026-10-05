from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import TypeAlias, cast, final

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from prejectory.core.errors import ManifestCompatibilityError
from prejectory.core.optional import raise_missing_optional_dependency
from prejectory.io.base import DatasetReader, split_directory_name
from prejectory.io.records import SceneRecord

try:
    import zarr
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The Zarr scene reader", extra="zarr")

ZARR_FORMAT_ID = "prejectory.scene.zarr"
ZARR_FORMAT_VERSION = 2

_NONE_I32 = -1
_NONE_I64 = np.iinfo(np.int64).min
_MAX_CACHED_SHARD_BYTES = 256 * 1024 * 1024

# scene/index columns; these must match zarr_writer.py.
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

_REQUIRED_ARRAYS = (
    "scene/index",
    "scene/position_offset",
    "agent/ids",
    "agent/types",
    "agent/screened_mask",
    "agent/features",
    "agent/valid_mask",
    "map/node_positions",
    "map/node_types",
    "map/edges",
)

ZarrArray: TypeAlias = zarr.Array[ArrayV2Metadata] | zarr.Array[ArrayV3Metadata]


@dataclass(frozen=True, slots=True)
class _ShardInfo:
    path: Path
    length: int


class ZarrReader(DatasetReader[SceneRecord]):
    """Read canonical scene records from all Zarr shards in one split."""

    def __init__(self, path: str | Path, split: str | None = None) -> None:
        self._path: Path = Path(path) / split_directory_name(split)
        if not self._path.exists():
            msg = f"Dataset split directory does not exist: {self._path}"
            raise FileNotFoundError(msg)
        if not self._path.is_dir():
            msg = f"Dataset split path is not a directory: {self._path}"
            raise NotADirectoryError(msg)

        self._shards: tuple[_ShardInfo, ...] = tuple(
            _read_shard_info(shard_path) for shard_path in sorted(self._path.glob("part-*.zarr"))
        )
        self._ends: tuple[int, ...] = tuple(accumulate(shard.length for shard in self._shards))

        # Shards are opened lazily. Constructing a split reader therefore only
        # reads each shard's root metadata, not all array objects/data.
        self._readers: OrderedDict[int, _ZarrShardReader] = OrderedDict()
        self._cached_capacity_bytes: int = 0

    @override
    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @override
    def __getitem__(self, at: int) -> SceneRecord:
        if at < 0:
            at += len(self)
        if not 0 <= at < len(self):
            raise IndexError(at)

        shard_index = bisect_right(self._ends, at)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return self._reader(shard_index)[at - start]

    def _reader(self, shard_index: int) -> _ZarrShardReader:
        reader = self._readers.get(shard_index)
        if reader is None:
            reader = _ZarrShardReader(self._shards[shard_index].path)
            while (
                self._readers
                and self._cached_capacity_bytes + reader.cache_capacity_bytes
                > _MAX_CACHED_SHARD_BYTES
            ):
                _, evicted = self._readers.popitem(last=False)
                self._cached_capacity_bytes -= evicted.cache_capacity_bytes
            self._readers[shard_index] = reader
            self._cached_capacity_bytes += reader.cache_capacity_bytes
        else:
            self._readers.move_to_end(shard_index)
        return reader


@final
class _LastChunk:
    """Keep a bounded set of decoded chunks for repeated scene reads."""

    def __init__(self, array: ZarrArray, *, axis: int = 0, max_chunks: int = 1) -> None:
        self._array = array
        self._axis = axis
        self._chunk_size = array.chunks[axis]
        self._max_chunks = max_chunks
        self._chunks: OrderedDict[int, npt.NDArray[np.generic]] = OrderedDict()

    @property
    def capacity_bytes(self) -> int:
        """Upper bound of decoded data retained by this cache."""
        return self._max_chunks * int(np.prod(self._array.chunks)) * self._array.dtype.itemsize

    def read(self, start: int, end: int) -> npt.NDArray[np.generic]:
        """Return an independent NumPy slice, including chunk-spanning slices."""
        if not 0 <= start <= end <= self._array.shape[self._axis]:
            msg = f"Invalid cached Zarr slice [{start}:{end}]"
            raise IndexError(msg)
        if start == end:
            shape = list(self._array.shape)
            shape[self._axis] = 0
            return np.empty(shape, dtype=self._array.dtype)

        pieces: list[npt.NDArray[np.generic]] = []
        while start < end:
            index = start // self._chunk_size
            data = self._chunks.get(index)
            if data is None:
                chunk_start = index * self._chunk_size
                chunk_end = min(chunk_start + self._chunk_size, self._array.shape[self._axis])
                if self._axis == 0:
                    data = np.asarray(self._array[chunk_start:chunk_end])
                else:
                    data = np.asarray(self._array[:, chunk_start:chunk_end])
                self._chunks[index] = data
                if len(self._chunks) > self._max_chunks:
                    _ = self._chunks.popitem(last=False)
            else:
                self._chunks.move_to_end(index)
            offset = start - index * self._chunk_size
            count = min(end - start, data.shape[self._axis] - offset)
            if self._axis == 0:
                pieces.append(data[offset : offset + count])
            else:
                pieces.append(data[:, offset : offset + count])
            start += count
        if len(pieces) == 1:
            return np.array(pieces[0], copy=True)
        return np.concatenate(pieces, axis=self._axis)


@final
class _ZarrShardReader(DatasetReader[SceneRecord]):
    def __init__(self, path: Path) -> None:
        self._path: Path = path
        # If consolidated metadata is present, Zarr uses it automatically.
        self._root: zarr.Group = zarr.open_group(path, mode="r")
        _validate_root(self._root, path)

        self._length = _require_nonnegative_int_attr(self._root, "record_count", path)
        self._arrays: dict[str, ZarrArray] = {
            name: self._require_array(name) for name in _REQUIRED_ARRAYS
        }
        self._validate_shapes()

        # These scene-sized columns are small and consulted for every record.
        # Materialize them once per opened shard instead of decoding their
        # chunks through Zarr for each individual scene.
        self._scene_index_data = np.asarray(self._arrays["scene/index"][:])
        self._position_offset_data = np.asarray(self._arrays["scene/position_offset"][:])
        self._features_cache = _LastChunk(self._arrays["agent/features"], max_chunks=2)
        self._agent_ids_cache = _LastChunk(self._arrays["agent/ids"], max_chunks=2)
        self._agent_types_cache = _LastChunk(self._arrays["agent/types"], max_chunks=2)
        self._screened_mask_cache = _LastChunk(self._arrays["agent/screened_mask"], max_chunks=2)
        self._valid_mask_cache = _LastChunk(self._arrays["agent/valid_mask"], max_chunks=2)
        self._map_node_positions_cache = _LastChunk(
            self._arrays["map/node_positions"], max_chunks=8
        )
        self._map_node_types_cache = _LastChunk(self._arrays["map/node_types"], max_chunks=8)
        self._map_edges_cache = _LastChunk(self._arrays["map/edges"], axis=1, max_chunks=8)
        self.cache_capacity_bytes = (
            self._scene_index_data.nbytes
            + self._position_offset_data.nbytes
            + sum(
                cache.capacity_bytes
                for cache in (
                    self._features_cache,
                    self._agent_ids_cache,
                    self._agent_types_cache,
                    self._screened_mask_cache,
                    self._valid_mask_cache,
                    self._map_node_positions_cache,
                    self._map_node_types_cache,
                    self._map_edges_cache,
                )
            )
        )

    @override
    def __len__(self) -> int:
        return self._length

    @override
    def __getitem__(self, at: int) -> SceneRecord:
        if at < 0:
            at += len(self)
        if not 0 <= at < len(self):
            raise IndexError(at)

        row = self._scene_index_data[at]

        dataset_id = int(row[_DATASET_ID])
        ego_agent_id = int(row[_EGO_AGENT_ID])
        prediction_origin = int(row[_PREDICTION_ORIGIN])
        prediction_end = int(row[_PREDICTION_END])

        agent_start = int(row[_AGENT_START])
        agent_end = int(row[_AGENT_END])
        node_start = int(row[_MAP_NODE_START])
        node_end = int(row[_MAP_NODE_END])
        edge_start = int(row[_MAP_EDGE_START])
        edge_end = int(row[_MAP_EDGE_END])

        edges = self._map_edges_cache.read(edge_start, edge_end)

        return SceneRecord(
            scene_number=int(row[_SCENE_NUMBER]),
            dataset_id=None if dataset_id == _NONE_I32 else dataset_id,
            ego_agent_id=None if ego_agent_id == _NONE_I64 else ego_agent_id,
            prediction_origin=None if prediction_origin == _NONE_I32 else prediction_origin,
            prediction_end=None if prediction_end == _NONE_I32 else prediction_end,
            position_offset=self._position_offset_data[at].copy(),
            agent_ids=cast(
                "npt.NDArray[np.int64]", self._agent_ids_cache.read(agent_start, agent_end)
            ),
            agent_types=cast(
                "npt.NDArray[np.int32]", self._agent_types_cache.read(agent_start, agent_end)
            ),
            screened_agent_mask=cast(
                "npt.NDArray[np.bool_]", self._screened_mask_cache.read(agent_start, agent_end)
            ),
            features=cast(
                "npt.NDArray[np.float32 | np.float64]",
                self._features_cache.read(agent_start, agent_end),
            ),
            valid_mask=cast(
                "npt.NDArray[np.bool_]", self._valid_mask_cache.read(agent_start, agent_end)
            ),
            map_node_positions=cast(
                "npt.NDArray[np.float32 | np.float64]",
                self._map_node_positions_cache.read(node_start, node_end),
            ),
            map_node_types=cast(
                "npt.NDArray[np.int32]", self._map_node_types_cache.read(node_start, node_end)
            ),
            map_edge_indices=cast("npt.NDArray[np.int32]", np.ascontiguousarray(edges[:2])),
            map_edge_types=np.ascontiguousarray(edges[2]),
        )

    def _require_array(self, name: str) -> ZarrArray:
        try:
            value = self._root[name]
        except KeyError as error:
            msg = f"Missing required Zarr array {name!r} in {self._path}."
            raise ValueError(msg) from error

        if isinstance(value, zarr.Group):
            msg = f"Expected Zarr array at {name!r} in {self._path}."
            raise TypeError(msg)
        return value

    def _validate_shapes(self) -> None:
        scene_index = self._arrays["scene/index"]
        if scene_index.ndim != 2 or scene_index.shape != (self._length, _SCENE_INDEX_WIDTH):
            msg = (
                f"Invalid scene/index shape in {self._path}: {scene_index.shape}; "
                f"expected ({self._length}, {_SCENE_INDEX_WIDTH})."
            )
            raise ValueError(msg)

        position_offset = self._arrays["scene/position_offset"]
        if position_offset.ndim != 2 or position_offset.shape != (self._length, 2):
            msg = (
                f"Invalid scene/position_offset shape in {self._path}: "
                f"{position_offset.shape}; expected ({self._length}, 2)."
            )
            raise ValueError(msg)

        map_edges = self._arrays["map/edges"]
        if map_edges.ndim != 2 or map_edges.shape[0] != 3:
            msg = f"Invalid map/edges shape in {self._path}: {map_edges.shape}; expected (3, E)."
            raise ValueError(msg)


def _read_shard_info(path: Path) -> _ShardInfo:
    # Root metadata alone is enough to route global indices to shards.
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    _validate_root(root, path)
    length = _require_nonnegative_int_attr(root, "record_count", path)
    return _ShardInfo(path=path, length=length)


def _validate_root(root: zarr.Group, path: Path) -> None:
    if root.attrs.get("format") != ZARR_FORMAT_ID:
        msg = f"Unsupported Zarr payload format in {path}."
        raise ValueError(msg)

    raw_version = root.attrs.get("format_version", 0)
    if not isinstance(raw_version, int):
        msg = f"Invalid Zarr format version in {path}."
        raise TypeError(msg)
    if raw_version != ZARR_FORMAT_VERSION:
        raise ManifestCompatibilityError(raw_version, ZARR_FORMAT_VERSION)

    raw_columns = root.attrs.get("scene_index_columns")
    if not isinstance(raw_columns, list) or tuple(raw_columns) != _SCENE_INDEX_COLUMNS:
        msg = f"Invalid scene/index column schema in {path}."
        raise ValueError(msg)


def _require_nonnegative_int_attr(root: zarr.Group, name: str, path: Path) -> int:
    value = root.attrs.get(name)
    if not isinstance(value, int):
        msg = f"Missing or invalid {name!r} attribute in {path}."
        raise TypeError(msg)
    if value < 0:
        msg = f"Invalid negative {name!r} attribute in {path}: {value}."
        raise ValueError(msg)
    return value
