"""Loader implementation for the I-80 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import POSITIONS_ONLY
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema


class NGSimLoader(SceneLoader[Path, NoLoaderOptions]):
    """Scene loader for the I-80 dataset."""

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path]]:
        for i, csv_file in enumerate(sorted(self.root.rglob("trajectories*.csv"))):
            yield DatasetSource(identifier=i, payload=csv_file)

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        yield LoadedSourceFrame(
            pl.scan_csv(source.payload).select(
                pl.col("Vehicle_ID").alias("id"),
                pl.col("Frame_ID").alias("frame"),
                pl.col("Local_X").alias("x").mul(0.3048),
                pl.col("Local_Y").alias("y").mul(0.3048),
                pl
                .col("v_Class")
                .replace_strict({
                    1: AgentCategory.MOTORCYCLE,
                    2: AgentCategory.CAR,
                    3: AgentCategory.TRUCK,
                })
                .alias("agent_category"),
                pl.col("Lane_ID").alias("lane_id"),
            ),
        )

    @override
    def count_sources(self) -> int | None:
        return sum(1 for path in self.root.rglob("trajectories*.csv") if path.is_file())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY
