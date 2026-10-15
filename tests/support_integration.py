from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl

from prejectory.datasets.registry import dataset_id_for_name
from prejectory.io.base import WorkerWriterProvider
from prejectory.io.encoding.common import encode_scene_record
from prejectory.runtime.executor import open_executor

if TYPE_CHECKING:
    from pathlib import Path

    from prejectory.core.maps import MapGraph
    from prejectory.core.scene import Scene
    from prejectory.core.scene.schema import TrajectorySchema
    from prejectory.io import SceneRecord
    from prejectory.runtime.state import Progress
    from prejectory.runtime.types import ExecutionPlan


def _assert_shape(array: npt.NDArray[Any], expected: tuple[int, ...], name: str) -> None:
    assert array.shape == expected, f"{name} must have shape {expected}, got {array.shape}"


def _assert_ndim(array: npt.NDArray[Any], expected: int, name: str) -> None:
    assert array.ndim == expected, f"{name} must be {expected}D, got {array.ndim}D"


def _assert_dtype(array: npt.NDArray[Any], expected: npt.DTypeLike, name: str) -> None:
    assert array.dtype == np.dtype(expected), (
        f"{name} must have dtype {expected}, got {array.dtype}"
    )


def _assert_finite(array: npt.NDArray[np.floating[Any]], name: str) -> None:
    assert np.isfinite(array).all(), f"{name} contains NaN/Inf"


def _assert_edge_index_bounds(
    edge_indices: npt.NDArray[np.int32],
    num_nodes: int,
    *,
    name: str,
) -> None:
    if edge_indices.shape[1] == 0:
        return
    assert num_nodes > 0, f"{name} has edges but no nodes"
    assert int(edge_indices.min()) >= 0, f"{name} contains negative node index"
    assert int(edge_indices.max()) < num_nodes, f"{name} contains out-of-bounds node index"


def assert_basic_scene_sanity(scene: Scene) -> None:
    assert scene.scene_number >= 0, "scene_number must be non-negative"
    assert scene.horizon_frames > 0, "expected positive total horizon"

    frame = scene.frame
    assert frame.height > 0, "scene frame is empty"

    required_columns = ("frame", "id", "x", "y", "agent_category")
    missing = [col for col in required_columns if col not in frame.columns]
    assert not missing, f"missing required columns: {missing}"

    null_counts = frame.select([
        pl.col(col).is_null().sum().alias(col) for col in required_columns
    ]).row(0, named=True)
    assert all(count == 0 for count in null_counts.values()), (
        f"nulls found in required columns: {null_counts}"
    )

    invalid_xy = frame.filter(~pl.col("x").is_finite() | ~pl.col("y").is_finite())
    assert invalid_xy.is_empty(), "found non-finite coordinates"

    assert int(frame["id"].n_unique()) > 0, "no agents found"
    assert int(frame["frame"].n_unique()) > 0, "no frames found"
    if frame.height > 1:
        x_span, y_span = frame.select(
            (pl.col("x").max() - pl.col("x").min()).alias("x_span"),
            (pl.col("y").max() - pl.col("y").min()).alias("y_span"),
        ).row(0)
        assert float(x_span) > 0.0 or float(y_span) > 0.0, (
            "degenerate scene with zero coordinate span"
        )

    duplicate_frames = frame.group_by("id", "frame").agg(count=pl.len()).filter(pl.col("count") > 1)
    assert duplicate_frames.is_empty(), "found duplicate frames for some agents"


def assert_basic_map_sanity(graph: MapGraph | None, *, expect_map: bool) -> None:
    if not expect_map:
        if graph is None:
            return
        assert graph.num_nodes == 0, "expected no map nodes for non-map scene"
        assert graph.num_edges == 0, "expected no map edges for non-map scene"
        return

    assert graph is not None, "scene advertises map support but resolved map is None"

    _assert_ndim(graph.node_positions, 2, "map node_positions")
    _assert_ndim(graph.edge_indices, 2, "map edge_indices")
    _assert_ndim(graph.node_types, 1, "map node_types")
    _assert_ndim(graph.edge_types, 1, "map edge_types")

    assert graph.node_positions.shape[1] == 2, "map node_positions must have shape (N, 2)"
    assert graph.edge_indices.shape[0] == 2, "map edge_indices must have shape (2, E)"

    assert graph.node_positions.shape[0] == graph.node_types.shape[0], (
        "map node count mismatch between node_positions and node_types"
    )
    assert graph.edge_indices.shape[1] == graph.edge_types.shape[0], (
        "map edge count mismatch between edge_indices and edge_types"
    )

    _assert_dtype(graph.node_positions, np.float64, "map node_positions")
    _assert_dtype(graph.edge_indices, np.int32, "map edge_indices")
    _assert_dtype(graph.node_types, np.int32, "map node_types")
    _assert_dtype(graph.edge_types, np.int32, "map edge_types")

    if graph.num_nodes > 0:
        _assert_finite(graph.node_positions, "map node_positions")

    _assert_edge_index_bounds(graph.edge_indices, graph.num_nodes, name="map edge_indices")


def assert_record_sanity(record: SceneRecord, scene: Scene) -> None:
    assert record.scene_number == scene.scene_number, "scene_number mismatch"
    assert record.dataset_id == _expected_dataset_id(scene.dataset), "dataset_id mismatch"

    _assert_shape(record.position_offset, (2,), "position_offset")
    _assert_finite(record.position_offset, "position_offset")

    n_agents = int(record.agent_types.shape[0])
    horizon_frames = scene.horizon_frames
    feature_dim = int(record.features.shape[2])

    _assert_shape(record.agent_ids, (n_agents,), "agent_ids")
    _assert_dtype(record.agent_ids, np.int64, "agent_ids")
    _assert_shape(record.screened_agent_mask, (n_agents,), "screened_agent_mask")
    _assert_dtype(record.screened_agent_mask, np.bool_, "screened_agent_mask")

    _assert_ndim(record.features, 3, "features")
    _assert_ndim(record.valid_mask, 2, "mask")

    assert record.features.shape[0] == n_agents, "features agent dim mismatch"
    assert record.valid_mask.shape[0] == n_agents, "mask agent dim mismatch"

    assert record.features.shape[1] == horizon_frames, "features time dim must match scene horizon"
    assert record.valid_mask.shape[1] == horizon_frames, "mask time dim must match scene horizon"

    assert feature_dim > 0, "feature dimension must be > 0"

    _assert_dtype(record.valid_mask, np.bool_, "mask")

    if record.valid_mask.any():
        assert np.isfinite(record.features[record.valid_mask]).all(), (
            "features contain NaN/Inf at valid mask positions"
        )

    if n_agents > 0:
        assert record.screened_agent_mask.any(), "no passing agents in encoded record"

    map_num_nodes = record.map_node_positions.shape[0]
    map_num_edges = record.map_edge_indices.shape[1]

    _assert_ndim(record.map_node_positions, 2, "map_node_positions")
    _assert_ndim(record.map_edge_indices, 2, "map_edge_indices")
    assert record.map_node_positions.shape[1] == 2, "map_node_positions must have shape (N, 2)"
    assert record.map_edge_indices.shape[0] == 2, "map_edge_indices must have shape (2, E)"
    _assert_shape(record.map_node_types, (map_num_nodes,), "map_node_types")
    _assert_shape(record.map_edge_types, (map_num_edges,), "map_edge_types")

    _assert_finite(record.map_node_positions, "map_node_positions")
    _assert_edge_index_bounds(record.map_edge_indices, map_num_nodes, name="map_edge_indices")

    has_record_map = map_num_nodes > 0 or map_num_edges > 0
    if not scene.has_map():
        assert not has_record_map, "encoded map payload present for scene without map"


def _expected_dataset_id(dataset: str | None) -> int | None:
    if dataset is None:
        return None
    return dataset_id_for_name(dataset)


@dataclass(frozen=True, slots=True)
class PlanSceneAssertionResult:
    checked_scenes: int
    written_scenes: int
    progress: Progress


class AssertingSceneWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        trajectory_schema: TrajectorySchema,
        artifact_dir: Path | None = None,
        dataset_name: str,
        scene_start: int = 0,
        scene_step: int = 1,
    ) -> None:
        self._marker_dir: Path = output_dir / ".integration-scene-assertions"
        self._marker_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_schema: TrajectorySchema = trajectory_schema
        self._artifact_dir: Path | None = artifact_dir
        self._dataset_name: str = dataset_name
        self._scene_start: int = scene_start
        self._scene_step: int = scene_step

    def write(self, scene: Scene) -> None:
        if scene.scene_number < self._scene_start:
            return
        if (scene.scene_number - self._scene_start) % self._scene_step != 0:
            return

        assert_basic_scene_sanity(scene)
        graph = scene.resolve_map()
        assert_basic_map_sanity(graph, expect_map=scene.has_map())
        record = encode_scene_record(
            scene,
            dtype=np.float32,
            trajectory_schema=self._trajectory_schema,
        )
        assert_record_sanity(record, scene)
        if self._artifact_dir is not None:
            save_scene_artifacts(
                scene=scene,
                graph=graph,
                out_dir=self._artifact_dir / f"scene_{scene.scene_number:03d}",
                dataset_name=self._dataset_name,
            )

        marker_path = self._marker_dir / f"{scene.scene_number:06d}.json"
        with marker_path.open("w", encoding="utf-8") as file_handle:
            json.dump({"scene_number": scene.scene_number}, file_handle)

    def finish_local(self) -> None: ...

    def flush_local(self) -> None: ...


def _create_asserting_scene_writer(
    _worker_id: int,
    *,
    output_dir: Path,
    trajectory_schema: TrajectorySchema,
    artifact_dir: Path | None,
    dataset_name: str,
    scene_start: int,
    scene_step: int,
) -> AssertingSceneWriter:
    return AssertingSceneWriter(
        output_dir=output_dir,
        trajectory_schema=trajectory_schema,
        artifact_dir=artifact_dir,
        dataset_name=dataset_name,
        scene_start=scene_start,
        scene_step=scene_step,
    )


def assert_plan_scene_outputs(
    plan: ExecutionPlan,
    *,
    artifact_dir: Path | None = None,
    dataset_name: str,
    scene_start: int = 0,
    scene_step: int = 1,
) -> PlanSceneAssertionResult:
    writer_provider = WorkerWriterProvider(
        create_worker=functools.partial(
            _create_asserting_scene_writer,
            output_dir=plan.output_dir,
            trajectory_schema=plan.trajectory_schema,
            artifact_dir=artifact_dir,
            dataset_name=dataset_name,
            scene_start=scene_start,
            scene_step=scene_step,
        ),
    )
    with open_executor(plan) as executor:
        progress = executor.execute(writer_provider)

    checked_scenes = sum(
        1 for _ in (plan.output_dir / ".integration-scene-assertions").glob("*.json")
    )
    return PlanSceneAssertionResult(
        checked_scenes=checked_scenes,
        written_scenes=progress.stats.written_scenes,
        progress=progress,
    )


def save_scene_artifacts(
    scene: Scene,
    graph: MapGraph | None,
    out_dir: Path,
    dataset_name: str,
) -> None:
    """Save scene artifacts like trajectories and maps for debugging."""

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import prejectory_viz as dviz  # pyright: ignore[reportMissingImports]  # ruff: ignore[import-outside-top-level]

        dviz.trajectory_figure(scene).save(out_dir / "trajectory.html")

    except ImportError:
        ...
    frame = scene.frame.select("frame", "id", "x", "y")
    summary_frame = frame.select(
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        pl.col("frame").min().alias("frame_min"),
        pl.col("frame").max().alias("frame_max"),
    )
    x_min, x_max, y_min, y_max, frame_min, frame_max = summary_frame.row(0)

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "scene_number": scene.scene_number,
        "rows": frame.height,
        "agents": int(frame["id"].n_unique()),
        "frame_min": int(frame_min),
        "frame_max": int(frame_max),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "has_map": graph is not None,
        "map_num_nodes": int(graph.num_nodes) if graph is not None else 0,
        "map_num_edges": int(graph.num_edges) if graph is not None else 0,
    }

    if graph is not None and graph.num_nodes > 0:
        summary["map_x_min"] = float(np.min(graph.node_positions[:, 0]))
        summary["map_x_max"] = float(np.max(graph.node_positions[:, 0]))
        summary["map_y_min"] = float(np.min(graph.node_positions[:, 1]))
        summary["map_y_max"] = float(np.max(graph.node_positions[:, 1]))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, sort_keys=True)
