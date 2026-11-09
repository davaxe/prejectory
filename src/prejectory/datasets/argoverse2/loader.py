"""Loader implementation for the Argoverse 2 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl
from pydantic import Field

from prejectory.core.categories import AgentCategory, DatasetSplit
from prejectory.core.scene import POSITIONS_VELOCITY_YAW
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import (
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
)
from prejectory.processing.maps import MapReference

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class Argoverse2LoaderOptions(LoaderOptionsModel):
    """Dataset-owned config for the Argoverse 2 loader."""

    file_batch_size: int = Field(default=100, ge=1)


class Argoverse2Loader(SceneLoader[list[Path], Argoverse2LoaderOptions]):
    """Loader for Argoverse 2 trajectory data stored in Parquet files."""

    def _sources_from_dir(self, data_dir: Path) -> Iterable[DatasetSource[list[Path]]]:
        if not data_dir.is_dir():
            return
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        for i in range(0, len(parquet_files), self.loader_options.file_batch_size):
            yield DatasetSource(
                identifier=i,
                payload=parquet_files[i : i + self.loader_options.file_batch_size],
            )

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[list[Path]]]:
        yield from self._sources_from_dir(self.root / split.value)

    @override
    def load_source(self, source: DatasetSource[list[Path]]) -> Iterable[LoadedSourceFrame]:
        file_to_map: dict[str, str] = {}
        for pq in source.payload:
            json_candidates = list(pq.parent.glob("*.json"))
            if json_candidates:
                file_to_map[str(pq)] = str(json_candidates[0])

        batch_lf = pl.scan_parquet(
            source.payload,
            include_file_paths="file_id",
            schema=_SCHEMA,
            extra_columns="ignore",
            cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
        ).select(
            pl.col("file_id"),
            self._map_object_type_expr("object_type").alias("agent_category"),
            pl.col("track_id").str.replace("AV", "0").cast(pl.Int32).alias("id"),
            pl.col("timestep").alias("frame").cast(pl.Int64),
            pl.col("position_x").alias("x"),
            pl.col("position_y").alias("y"),
            pl.col("velocity_x").alias("vx"),
            pl.col("velocity_y").alias("vy"),
            pl.col("heading").alias("yaw"),
        )

        for (file_id,), group in batch_lf.collect().group_by(["file_id"]):
            yield LoadedSourceFrame(
                frame=group.lazy().drop("file_id"),
                map_reference=MapReference(map_key=file_to_map.get(str(file_id))),
                ego_agent_id=0,
            )

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "val")
        return self._count_sources(self.root / "test")

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @staticmethod
    def _map_object_type_expr(col: str) -> pl.Expr:
        mapping = {
            "static": AgentCategory.STATIC_OBJECT,
            "riderless_bicycle": AgentCategory.STATIC_OBJECT,
            "construction": AgentCategory.STATIC_OBJECT,
            "vehicle": AgentCategory.CAR,
            "motorcyclist": AgentCategory.MOTORCYCLE,
            "cyclist": AgentCategory.BICYCLE,
            "bus": AgentCategory.BUS,
            "pedestrian": AgentCategory.PEDESTRIAN,
            "background": AgentCategory.UNIMPORTANT,
            "unknown": AgentCategory.UNKNOWN,
        }
        return pl.col(col).replace_strict(mapping, return_dtype=pl.Int32)

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*/*.parquet"))
        batches, extra = divmod(num_files, self.loader_options.file_batch_size)
        return batches + int(extra > 0)


_SCHEMA: pl.Schema = pl.Schema({
    "object_type": pl.Utf8,
    "track_id": pl.Utf8,
    "timestep": pl.Float64,
    "position_x": pl.Float64,
    "position_y": pl.Float64,
    "velocity_x": pl.Float64,
    "velocity_y": pl.Float64,
    "heading": pl.Float64,
})
