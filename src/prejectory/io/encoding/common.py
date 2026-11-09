"""Backend-independent encoding helpers for scene records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import TypedDict

from prejectory.core.categories import AgentCategory
from prejectory.datasets.registry import dataset_id_for_name
from prejectory.io.records import ForecastRecord, PredictionBounds, SceneRecord

if TYPE_CHECKING:
    from prejectory.core.maps import MapGraph
    from prejectory.core.scene import Scene, TrajectorySchema
    from prejectory.core.typing import FloatDType


class MapRecord[FloatScalarT: (np.float32, np.float64)](TypedDict):
    """Simple NumPy representation of one scene's map payload."""

    map_node_positions: npt.NDArray[FloatScalarT]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


MapRecordF32 = MapRecord[np.float32]
MapRecordF64 = MapRecord[np.float64]
AnyMapRecord = MapRecord[np.float32] | MapRecord[np.float64]


def encode_scene_record(
    scene: Scene,
    *,
    dtype: FloatDType,
    recenter_position: bool = True,
    trajectory_schema: TrajectorySchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
    prediction_bounds: PredictionBounds | None = None,
    dataset_id: int | None = None,
) -> SceneRecord:
    """Encode one scene into the canonical full-horizon `SceneRecord`."""
    if trajectory_schema is not None:
        scene = scene.as_schema(trajectory_schema)

    data = scene.frame
    feature_columns = scene.schema.feature_columns()
    time_steps = scene.horizon_frames
    start_frame = data["frame"].min()
    unique_ids: list[int] = data["id"].unique().to_list()
    sorted_ids = sorted(unique_ids)
    num_agents = len(unique_ids)
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted_ids)}
    df_indexed = data.with_columns([
        (pl.col("frame") - start_frame).alias("col_idx"),
        pl.col("id").replace_strict(id_to_idx_map).alias("row_idx"),
    ])

    if recenter_position:
        mean_x = data["x"].mean()
        mean_y = data["y"].mean()
        offset = np.array([mean_x, mean_y], dtype=np.float64)
    else:
        offset = np.zeros((2,), dtype=np.float64)

    row_indices = df_indexed["row_idx"].to_numpy()
    col_indices = df_indexed["col_idx"].to_numpy()

    features = np.zeros((num_agents, time_steps, len(feature_columns)), dtype=dtype)
    mask = np.zeros((num_agents, time_steps), dtype=bool)

    for feature_idx, column in enumerate(feature_columns):
        values = df_indexed[column].to_numpy(writable=column in {"x", "y"})
        if column == "x":
            values -= offset[0]
        elif column == "y":
            values -= offset[1]
        features[row_indices, col_indices, feature_idx] = values.astype(dtype, copy=False)

    mask[row_indices, col_indices] = True
    type_df = df_indexed.unique(subset="row_idx", keep="first").sort("row_idx")

    raw_categories = type_df["agent_category"].to_numpy()
    if category_mapping:
        agent_types = np.array(
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        agent_types = raw_categories.astype(np.int32)

    screened_agent_mask = np.ones((num_agents,), dtype=bool)
    if scene.passed_agent_ids is not None:
        screened_agent_mask = np.array(
            [agent_id in scene.passed_agent_ids for agent_id in sorted_ids],
            dtype=bool,
        )

    map_record = encode_map_from_scene(scene, dtype=dtype, offset=offset)
    return SceneRecord(
        scene_number=scene.scene_number,
        position_offset=offset,
        agent_ids=np.asarray(sorted_ids, dtype=np.int64),
        agent_types=agent_types,
        screened_agent_mask=screened_agent_mask,
        features=features,
        valid_mask=mask,
        map_node_positions=map_record["map_node_positions"],
        map_edge_indices=map_record["map_edge_indices"],
        map_node_types=map_record["map_node_types"],
        map_edge_types=map_record["map_edge_types"],
        dataset_id=_resolve_dataset_id(scene.dataset, explicit_dataset_id=dataset_id),
        prediction_origin=(
            None if prediction_bounds is None else prediction_bounds.prediction_origin
        ),
        prediction_end=None if prediction_bounds is None else prediction_bounds.prediction_end,
        ego_agent_id=scene.ego_agent_id,
    )


def _resolve_dataset_id(dataset: str | None, *, explicit_dataset_id: int | None) -> int | None:
    if explicit_dataset_id is not None:
        return explicit_dataset_id
    if dataset is None:
        return None
    return dataset_id_for_name(dataset)


def encode_split_scene_record(
    scene: Scene,
    *,
    dtype: FloatDType,
    prediction_origin: int,
    prediction_end: int | None = None,
    recenter_position: bool = True,
    trajectory_schema: TrajectorySchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> ForecastRecord:
    """Encode one scene and split it into observation/prediction tensors."""
    record = encode_scene_record(
        scene,
        dtype=dtype,
        recenter_position=recenter_position,
        trajectory_schema=trajectory_schema,
        category_mapping=category_mapping,
    )
    return record.forecast(
        PredictionBounds(
            prediction_origin, record.horizon_frames if prediction_end is None else prediction_end
        )
    )


def empty_map_record(dtype: FloatDType) -> MapRecordF32 | MapRecordF64:
    """Return the canonical empty-map payload used by raw scene records."""
    map_dict: MapRecord[Any] = {
        "map_node_positions": np.empty((0, 2), dtype=dtype),
        "map_edge_indices": np.empty((2, 0), dtype=np.int32),
        "map_node_types": np.empty((0,), dtype=np.int32),
        "map_edge_types": np.empty((0,), dtype=np.int32),
    }
    return map_dict


def encode_map_from_scene(
    scene: Scene,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None = None,
) -> MapRecordF32 | MapRecordF64:
    """Resolve one scene map and convert it to the canonical raw-record layout."""
    graph = scene.resolve_map()
    if graph is None:
        return empty_map_record(dtype)

    return _map_graph_to_numpy(graph, dtype=dtype, offset=offset)


def _map_graph_to_numpy(
    graph: MapGraph,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None = None,
) -> MapRecordF32 | MapRecordF64:
    """Convert a `MapGraph` to one canonical NumPy map payload."""
    node_positions = graph.node_positions - offset if offset is not None else graph.node_positions
    map_dict: MapRecord[Any] = {
        "map_node_positions": node_positions.astype(dtype, copy=False),
        "map_edge_indices": np.ascontiguousarray(graph.edge_indices, dtype=np.int32),
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
    return map_dict
