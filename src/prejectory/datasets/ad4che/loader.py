"""Loader implementation for the AD4CHE dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import POSITIONS_VELOCITY_ACCELERATION
from prejectory.datasets.levelx.loader import LevelXDataLoader, LevelXSourceData
from prejectory.processing.loading.models import DatasetSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema
    from prejectory.processing.maps import MapProvider
    from prejectory.processing.models import LoaderPlan


class AD4CHELoader(LevelXDataLoader):
    """Loader for the AD4CHE dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        super().__init__(
            data_root=Path(data_root) / "AD4CHE_Data_V1.0",
            request=request,
            map_provider=map_provider,
        )

    @override
    def iter_sources(self) -> Iterable[DatasetSource[LevelXSourceData]]:
        for recording_id, subdir in self._recordings():
            yield DatasetSource(
                identifier=recording_id,
                payload=LevelXSourceData(path=subdir),
                map_key=f"{subdir.name}/{recording_id:02d}_laneWidthColorAndID.png",
            )

    @override
    def count_sources(self) -> int | None:
        return sum(1 for _ in self._recordings())

    @staticmethod
    @override
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl
            .col("class")
            .replace_strict({
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "bus": AgentCategory.BUS.value,
            })
            .alias("agent_category"),
        ]

    @staticmethod
    @override
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        return [
            pl.col("frame"),
            pl.col("id"),
            pl.col("x").add(pl.col("width") / 2),
            pl.col("y").add(pl.col("height") / 2),
            pl.col("xVelocity").alias("vx"),
            pl.col("yVelocity").alias("vy"),
            pl.col("xAcceleration").alias("ax"),
            pl.col("yAcceleration").alias("ay"),
            pl.col("laneId").alias("lane_id"),
        ]

    @staticmethod
    @override
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_ACCELERATION

    def _recordings(self) -> Iterable[tuple[int, Path]]:
        """Yield discovered recording identifiers with their directories."""
        for subdir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            number_str = subdir.name.split("_")[-1]
            yield int(number_str), subdir


_META_SCHEMA: pl.Schema = pl.Schema({"id": pl.Int32, "numLaneChanges": pl.Int32})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "id": pl.Int32,
    "width": pl.Float64,
    "height": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
})
