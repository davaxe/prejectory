"""Loader implementation for the ApolloScape dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import POSITIONS_YAW
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema


class ApolloScapeLoader(SceneLoader[Path, NoLoaderOptions]):
    """Loader for ApolloScape prediction trajectories."""

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path]]:
        for datafile in (self.root / "prediction_train").glob("*.txt"):
            yield DatasetSource(identifier=datafile.stem, payload=datafile)

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        yield LoadedSourceFrame(
            pl.scan_csv(
                source.payload,
                has_header=False,
                schema=_DATA_SCHEMA,
                separator=" ",
            ).select(
                *("frame", "id", "x", "y", "yaw"),
                pl.col("agent_category").replace_strict({
                    1: AgentCategory.CAR.value,
                    2: AgentCategory.TRUCK.value,
                    3: AgentCategory.PEDESTRIAN.value,
                    4: AgentCategory.BICYCLE.value,
                    5: AgentCategory.UNKNOWN.value,
                }),
            ),
        )

    @override
    def count_sources(self) -> int | None:
        return sum(1 for _ in (self.root / "prediction_train").glob("*.txt"))

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_YAW


_DATA_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int64,
    "id": pl.Int64,
    "agent_category": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "yaw": pl.Float64,
})
