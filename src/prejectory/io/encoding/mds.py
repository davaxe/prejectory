"""MDS-specific scene-record serialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict

from prejectory.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Mapping


class MDSRow(TypedDict):
    """Serialized MDS row for one scene record."""

    scene_number: int
    dataset_id: int
    prediction_origin: int
    prediction_end: int
    ego_agent_id: int
    position_offset: npt.NDArray[np.float64]
    agent_ids: npt.NDArray[np.int64]
    agent_types: npt.NDArray[np.int32]
    screened_agent_mask: npt.NDArray[np.uint8]
    features: npt.NDArray[np.float32 | np.float64]
    mask: npt.NDArray[np.uint8]
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


def encode_mds_row(record: SceneRecord) -> MDSRow:
    """Convert one scene record to the MDS payload layout."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _encode_mds_map_arrays(
        record.map_node_positions,
        record.map_edge_indices,
        record.map_node_types,
        record.map_edge_types,
    )
    return {
        "scene_number": int(record.scene_number),
        "dataset_id": -1 if record.dataset_id is None else int(record.dataset_id),
        "prediction_origin": -1 if record.prediction_origin is None else record.prediction_origin,
        "prediction_end": -1 if record.prediction_end is None else record.prediction_end,
        "ego_agent_id": -1 if record.ego_agent_id is None else int(record.ego_agent_id),
        "position_offset": record.position_offset,
        "agent_ids": record.agent_ids,
        "agent_types": record.agent_types,
        "screened_agent_mask": record.screened_agent_mask.astype(np.uint8, copy=False),
        "features": record.features,
        "mask": record.valid_mask.astype(np.uint8, copy=False),
        "map_node_positions": map_node_positions,
        "map_edge_indices": map_edge_indices,
        "map_node_types": map_node_types,
        "map_edge_types": map_edge_types,
    }


def decode_mds_row(row: Mapping[str, Any]) -> SceneRecord:
    """Convert one MDS row into the canonical scene record."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _decode_mds_map_arrays(
        np.asarray(row["map_node_positions"]),
        np.asarray(row["map_edge_indices"]),
        np.asarray(row["map_node_types"]),
        np.asarray(row["map_edge_types"]),
    )
    return SceneRecord(
        scene_number=int(row["scene_number"]),
        ego_agent_id=(None if int(row.get("ego_agent_id", -1)) < 0 else int(row["ego_agent_id"])),
        dataset_id=(None if int(row.get("dataset_id", -1)) < 0 else int(row["dataset_id"])),
        prediction_origin=(
            None if int(row.get("prediction_origin", -1)) < 0 else int(row["prediction_origin"])
        ),
        prediction_end=(
            None if int(row.get("prediction_end", -1)) < 0 else int(row["prediction_end"])
        ),
        position_offset=np.asarray(row["position_offset"], dtype=np.float64),
        agent_ids=np.asarray(row["agent_ids"], dtype=np.int64),
        agent_types=np.asarray(row["agent_types"], dtype=np.int32),
        screened_agent_mask=np.asarray(row["screened_agent_mask"], dtype=bool),
        features=cast("npt.NDArray[np.float32 | np.float64]", np.asarray(row["features"])),
        valid_mask=np.asarray(row["mask"], dtype=bool),
        map_node_positions=map_node_positions,
        map_edge_indices=map_edge_indices,
        map_node_types=map_node_types,
        map_edge_types=map_edge_types,
    )


def mds_columns(dtype: str) -> dict[str, str]:
    """Return the MDS column schema for one serialized scene record."""
    return {
        "scene_number": "int",
        "dataset_id": "int",
        "prediction_origin": "int",
        "prediction_end": "int",
        "ego_agent_id": "int",
        "position_offset": "ndarray:float64:2",
        "agent_ids": "ndarray:int64",
        "agent_types": "ndarray:int32",
        "screened_agent_mask": "ndarray:uint8",
        "features": f"ndarray:{dtype}",
        "mask": "ndarray:uint8",
        "map_node_positions": f"ndarray:{dtype}",
        "map_edge_indices": "ndarray:int32",
        "map_node_types": "ndarray:int32",
        "map_edge_types": "ndarray:int32",
    }


def _encode_mds_map_arrays(
    map_node_positions: npt.NDArray[np.float32 | np.float64],
    map_edge_indices: npt.NDArray[np.int32],
    map_node_types: npt.NDArray[np.int32],
    map_edge_types: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    if map_node_positions.size == 0:
        map_node_positions = np.full((1, 2), dtype=map_node_positions.dtype, fill_value=np.nan)
    if map_edge_indices.size == 0:
        map_edge_indices = np.full((2, 1), dtype=np.int32, fill_value=-1)
    if map_node_types.size == 0:
        map_node_types = np.full((1,), dtype=np.int32, fill_value=-1)
    if map_edge_types.size == 0:
        map_edge_types = np.full((1,), dtype=np.int32, fill_value=-1)
    return map_node_positions, map_edge_indices, map_node_types, map_edge_types


def _decode_mds_map_arrays(
    map_node_positions: npt.NDArray[Any],
    map_edge_indices: npt.NDArray[Any],
    map_node_types: npt.NDArray[Any],
    map_edge_types: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    if map_node_positions.shape == (1, 2) and np.isnan(map_node_positions).all():
        map_node_positions = np.empty((0, 2), dtype=map_node_positions.dtype)
    if map_edge_indices.shape == (2, 1) and (map_edge_indices == -1).all():
        map_edge_indices = np.empty((2, 0), dtype=map_edge_indices.dtype)
    if map_node_types.shape == (1,) and (map_node_types == -1).all():
        map_node_types = np.empty((0,), dtype=map_node_types.dtype)
    if map_edge_types.shape == (1,) and (map_edge_types == -1).all():
        map_edge_types = np.empty((0,), dtype=map_edge_types.dtype)
    return map_node_positions, map_edge_indices, map_node_types, map_edge_types
