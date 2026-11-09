"""Loader implementation for the Argoverse 1 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl
from pydantic import Field

from prejectory.core.categories import AgentCategory, DatasetSplit
from prejectory.core.scene import POSITIONS_ONLY
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import (
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
)
from prejectory.processing.maps import MapProvider, MapReference

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema
    from prejectory.processing.models import LoaderPlan


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class Argoverse1LoaderOptions(LoaderOptionsModel):
    """Dataset-owned config for the Argoverse 1 loader."""

    file_batch_size: int = Field(default=10, ge=1)


class Argoverse1Loader(SceneLoader[list[Path], Argoverse1LoaderOptions]):
    """Loader for Argoverse 1 forecasting trajectories."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, map_provider=map_provider)
        self._train_dir: Path = self.root / "forecasting_train_v1.1" / "train" / "data"
        self._val_dir: Path = self.root / "forecasting_val_v1.1" / "val" / "data"
        self._test_dir: Path = self.root / "forecasting_test_v1.1" / "test_obs" / "data"

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self._train_dir)
        elif split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self._val_dir)
        else:
            yield from self._sources_from_dir(self._test_dir)

    @override
    def load_source(self, source: DatasetSource[list[Path]]) -> Iterable[LoadedSourceFrame]:
        batch_lf = (
            pl
            .scan_csv(source.payload, include_file_paths="file_id", schema=_SCHEMA)
            .with_columns(
                pl
                .when(pl.col("OBJECT_TYPE") == "AV")
                .then(AgentCategory.CAR)
                .otherwise(AgentCategory.UNKNOWN)
                .alias("agent_category"),
                pl
                .col("TRACK_ID")
                .rank(method="dense")
                .over("file_id")
                .sub(1)
                .cast(pl.Int64)
                .alias("id"),
                pl
                .col("TIMESTAMP")
                .rank(method="dense")
                .over("file_id")
                .sub(1)
                .cast(pl.Int64)
                .alias("frame"),
                pl.col("file_id").cast(pl.Categorical).to_physical(),
            )
            .drop("TRACK_ID", "TIMESTAMP")
            .rename({"X": "x", "Y": "y", "CITY_NAME": "map"})
        )

        for _, group in batch_lf.collect().group_by(["file_id"]):
            ego_vehicle_id = int(
                group
                .filter(pl.col("agent_category") == AgentCategory.CAR)
                .select(pl.col("id").first())
                .item(),
            )
            yield LoadedSourceFrame(
                frame=group.drop("file_id", "OBJECT_TYPE").lazy(),
                map_reference=MapReference(map_key=str(group["map"].first())),
                ego_agent_id=ego_vehicle_id,
            )

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._train_dir)
        if split is DatasetSplit.VAL:
            return self._count_sources(self._val_dir)
        return self._count_sources(self._test_dir)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    def _sources_from_dir(self, data_dir: Path) -> Iterable[DatasetSource[list[Path]]]:
        if not data_dir.is_dir():
            return
        files = sorted(data_dir.glob("*.csv"))
        for start in range(0, len(files), self.loader_options.file_batch_size):
            yield DatasetSource(
                identifier=start,
                payload=files[start : start + self.loader_options.file_batch_size],
            )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*.csv"))
        batches, extra = divmod(num_files, self.loader_options.file_batch_size)
        return batches + int(extra > 0)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float64,
    "Y": pl.Float64,
    "CITY_NAME": pl.String,
})
