"""Loader implementation for the A43 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import POSITIONS_VELOCITY_ACCELERATION
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema


class A43Loader(SceneLoader[Path, NoLoaderOptions]):
    """Scene loader for the A43 dataset."""

    _dt: ClassVar[float] = 0.1
    _eps: ClassVar[float] = 1e-9

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path]]:
        for i, csv_file in enumerate(self.root.glob("*.csv")):
            yield DatasetSource(identifier=i, payload=csv_file, map_key=csv_file.stem)

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:

        path = source.payload
        yield LoadedSourceFrame(
            pl
            .scan_csv(path, schema=_SCHEMA)
            .with_columns(t0=pl.col("tseconds").min())
            .with_columns(
                frame=(
                    (((pl.col("tseconds") - pl.col("t0")) / self._dt) + 0.5 + self._eps)
                    .floor()
                    .cast(pl.Int64)
                ),
            )
            .select(
                pl.col("ID").alias("id"),
                pl.col("frame"),
                *("x", "y", "vy", "vx", "ax", "ay"),
                pl
                .col("VehicleCategory")
                .replace_strict({
                    "Motorcycle": AgentCategory.MOTORCYCLE,
                    "Passenger Car": AgentCategory.CAR,
                    "Semi-trailer truck": AgentCategory.TRUCK,
                    "Truck": AgentCategory.TRUCK,
                    "Van": AgentCategory.VAN,
                    "Bus": AgentCategory.BUS,
                })
                .alias("agent_category"),
            ),
        )

    @override
    def count_sources(self) -> int | None:
        return sum(1 for _ in self.iter_sources())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_ACCELERATION


_SCHEMA: pl.Schema = pl.Schema({
    "tseconds": pl.Float64,
    "ttimestamp": pl.Utf8,
    "ID": pl.Int64,
    "VehicleCategory": pl.Utf8,
    "Length": pl.Float64,
    "Width": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})
