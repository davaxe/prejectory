"""Loader implementation for the SinD dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.functional import yaw_from_position_expr
from prejectory.core.scene import CANONICAL
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.core.scene import TrajectorySchema


class SindLoader(SceneLoader):
    """Loader for the SinD dataset."""

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path]]:
        for region in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for data_dir in sorted(p for p in region.iterdir() if p.is_dir()):
                yield DatasetSource(
                    identifier=data_dir.name,
                    payload=data_dir,
                    map_key=str(region.name),
                )

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        subdir = source.payload
        pedestrian_data_path = subdir / "Ped_smoothed_tracks.csv"
        vehicle_data_path = subdir / "Veh_smoothed_tracks.csv"
        vehicle_df = pl.scan_csv(vehicle_data_path, schema_overrides=_VEHICLE_SCHEMA).select(
            pl.col("track_id").alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "motorcycle": AgentCategory.MOTORCYCLE.value,
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "tricycle": AgentCategory.TRICYCLE.value,
                "bus": AgentCategory.BUS.value,
                "bicycle": AgentCategory.BICYCLE.value,
            })
            .alias("agent_category"),
            pl.col("heading_rad").alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        pedestrian_df = pl.scan_csv(
            pedestrian_data_path,
            schema_overrides=_PEDESTRIAN_SCHEMA,
        ).select(
            pl
            .col("track_id")
            .str.slice(1)
            .cast(pl.Int32)
            .add(vehicle_df.select(pl.col("id").max()).collect().item() + 1)
            .alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "pedestrian": AgentCategory.PEDESTRIAN.value,
                "animal": AgentCategory.ANIMAL.value,
            })
            .alias("agent_category"),
            yaw_from_position_expr().alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        yield LoadedSourceFrame(pl.concat([vehicle_df, pedestrian_df]))

    @override
    def count_sources(self) -> int | None:
        if not self.root.is_dir():
            return 0
        return sum(1 for path in self.root.iterdir() if path.is_dir())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL


_VEHICLE_SCHEMA = pl.Schema({
    "track_id": pl.Int32,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "heading_rad": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})

_PEDESTRIAN_SCHEMA = pl.Schema({
    "track_id": pl.Utf8,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})
