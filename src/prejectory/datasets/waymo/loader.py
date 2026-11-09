"""Loader implementation for the Waymo Open Dataset."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory, DatasetSplit
from prejectory.core.scene import POSITIONS_VELOCITY_YAW
from prejectory.datasets.waymo.protos import lean_scenario_pb2
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame
from prejectory.processing.maps import MapReference

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.core.scene import TrajectorySchema


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class WaymoLoader(SceneLoader):
    """Loader for Waymo scenarios stored in TFRecord format."""

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[DatasetSource[Path]]:
        if not data_dir.is_dir():
            return
        for tfrecord_path in sorted(data_dir.glob("*.tfrecord*")):
            yield DatasetSource(identifier=tfrecord_path.stem, payload=tfrecord_path)

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[Path]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self.root / "training")
        elif split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self.root / "validation")
        else:
            yield from self._sources_from_dir(self.root / "testing")

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "training")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "validation")
        return self._count_sources(self.root / "testing")

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        for scenario_index, raw_data in enumerate(_read_tfrecord(source.payload)):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)
            yield LoadedSourceFrame(
                frame=_scenario_to_polars(scenario).lazy().with_columns(pl.col("id").add(1)),
                map_reference=MapReference(
                    map_key=f"{source.identifier}:{scenario_index}",
                    map_payload=raw_data if self.map_config is not None else None,
                ),
                ego_agent_id=0,
            )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @staticmethod
    def _count_sources(data_dir: Path) -> int:
        return sum(1 for _ in data_dir.glob("*.tfrecord*")) if data_dir.is_dir() else 0


def _scenario_to_polars(scenario: lean_scenario_pb2.LeanScenario) -> pl.DataFrame:
    ego_track_index = scenario.sdc_track_index
    l_frame: list[int] = []
    l_tid: list[int] = []
    l_x: list[float] = []
    l_y: list[float] = []
    l_vx: list[float] = []
    l_vy: list[float] = []
    l_yaw: list[float] = []
    l_cat: list[int] = []

    for i, track in enumerate(scenario.tracks):
        t_id = -1 if i == ego_track_index else track.id
        cat_val = _OBJECT_TYPE_TO_CATEGORY[track.object_type]
        for frame_idx, state in enumerate(track.states):
            if not state.valid:
                continue
            l_frame.append(frame_idx)
            l_tid.append(t_id)
            l_x.append(state.center_x)
            l_y.append(state.center_y)
            l_vx.append(state.velocity_x)
            l_vy.append(state.velocity_y)
            l_yaw.append(state.heading)
            l_cat.append(cat_val)

    return pl.DataFrame(
        {
            "frame": l_frame,
            "id": l_tid,
            "x": l_x,
            "y": l_y,
            "vx": l_vx,
            "vy": l_vy,
            "yaw": l_yaw,
            "agent_category": l_cat,
        },
        schema={
            "frame": pl.Int32,
            "id": pl.Int32,
            "x": pl.Float64,
            "y": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "yaw": pl.Float64,
            "agent_category": pl.Int32,
        },
    )


def _read_tfrecord(path: Path) -> Iterable[bytes]:
    data = path.read_bytes()
    offset = 0
    unpack_len = struct.Struct("<Q").unpack_from
    while offset < len(data):
        if offset + 8 > len(data):
            break
        record_len = unpack_len(data, offset)[0]
        data_start = offset + 12
        data_end = data_start + record_len
        yield data[data_start:data_end]
        offset = data_end + 4


_OBJECT_TYPE_TO_CATEGORY: dict[int, AgentCategory] = {
    0: AgentCategory.UNKNOWN,
    1: AgentCategory.CAR,
    2: AgentCategory.PEDESTRIAN,
    3: AgentCategory.BICYCLE,
    4: AgentCategory.UNKNOWN,
}
