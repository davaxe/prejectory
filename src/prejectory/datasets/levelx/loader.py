"""Loader implementations for LevelX-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import CANONICAL, POSITIONS_VELOCITY_ACCELERATION
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema
    from prejectory.processing.maps import MapProvider
    from prejectory.processing.models import LoaderPlan


@dataclass(slots=True, frozen=True)
class LevelXSourceData:
    path: Path
    x0: float | None = None
    y0: float | None = None


class LevelXDataLoader(SceneLoader[LevelXSourceData, NoLoaderOptions]):
    """Common recording CSV loader for LevelX-style datasets."""

    @staticmethod
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("trackId").alias("id"),
            pl
            .col("class")
            .replace_strict(
                {
                    "car": AgentCategory.CAR.value,
                    "truck": AgentCategory.TRUCK.value,
                    "bus": AgentCategory.BUS.value,
                    "trailer": AgentCategory.TRAILER.value,
                    "motorcycle": AgentCategory.MOTORCYCLE.value,
                    "bicycle": AgentCategory.BICYCLE.value,
                    "pedestrian": AgentCategory.PEDESTRIAN.value,
                    "van": AgentCategory.VAN.value,
                    "truck_bus": AgentCategory.TRUCK.value,
                    "animal": AgentCategory.ANIMAL.value,
                },
                return_dtype=pl.Int32,
            )
            .alias("agent_category"),
        ]

    @staticmethod
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        return [
            pl.col("frame"),
            pl.col("trackId").alias("id"),
            pl.col("xCenter").alias("x"),
            pl.col("yCenter").alias("y"),
            pl.col("xVelocity").alias("vx"),
            pl.col("yVelocity").alias("vy"),
            pl.col("xAcceleration").alias("ax"),
            pl.col("yAcceleration").alias("ay"),
            pl.col("heading").alias("yaw"),
        ]

    @staticmethod
    def location_id_select(meta_df: pl.DataFrame, path: Path) -> str:
        """Select the map key from the recording metadata CSV."""
        _ = path
        return meta_df.select(pl.col("locationId")).item()

    @staticmethod
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @override
    def iter_sources(self) -> Iterable[DatasetSource[LevelXSourceData]]:
        for recording_id in self._recording_ids():
            recording_meta = self.root / f"{recording_id:0>2}_recordingMeta.csv"
            recording_meta_data = pl.read_csv(recording_meta)
            location_id = self.location_id_select(recording_meta_data, recording_meta)
            columns = recording_meta_data.columns

            utm_x0: float | None = None
            utm_y0: float | None = None
            if "xUtmOrigin" in columns and "yUtmOrigin" in columns:
                utm_x0 = recording_meta_data.select(pl.col("xUtmOrigin")).item()
                utm_y0 = recording_meta_data.select(pl.col("yUtmOrigin")).item()

            yield DatasetSource(
                identifier=recording_id,
                payload=LevelXSourceData(self.root, x0=utm_x0, y0=utm_y0),
                map_key=str(location_id),
            )

    @override
    def load_source(self, source: DatasetSource[LevelXSourceData]) -> Iterable[LoadedSourceFrame]:
        tracks = source.payload.path / f"{source.identifier:0>2}_tracks.csv"
        meta = source.payload.path / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select(),
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select(),
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        combined = combined.with_columns(
            (pl.col("x") + (source.payload.x0 or 0.0)).alias("x"),
            (pl.col("y") + (source.payload.y0 or 0.0)).alias("y"),
        )
        yield LoadedSourceFrame(combined)

    @override
    def count_sources(self) -> int | None:
        return len(self._recording_ids())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL

    def _recording_ids(self) -> list[int]:
        """Return sorted recording identifiers discovered from metadata files."""
        recording_ids: list[int] = []
        for recording_meta in sorted(self.root.glob("*_recordingMeta.csv")):
            prefix, _, _ = recording_meta.stem.partition("_")
            recording_ids.append(int(prefix))
        return recording_ids


class StandardLevelXLoader(LevelXDataLoader):
    """Loader for LevelX datasets that store recordings under a `data` directory."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        super().__init__(
            data_root=Path(data_root) / "data",
            request=request,
            map_provider=map_provider,
        )


class ExiDLoader(StandardLevelXLoader):
    """Loader for the ExiD dataset."""

    @staticmethod
    @override
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        select = LevelXDataLoader.track_data_select()
        select.append(pl.col("laneletId").alias("lane_id"))
        return select

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _EXID_TRACK_SCHEMA


class HighDLoader(LevelXDataLoader):
    """Loader for the highD dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        super().__init__(
            data_root=Path(data_root) / "data",
            request=request,
            map_provider=map_provider,
        )

    @staticmethod
    @override
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl.col("numLaneChanges").alias("lane_changes"),
            pl
            .col("class")
            .replace_strict(
                {"Car": AgentCategory.CAR.value, "Truck": AgentCategory.TRUCK.value},
                return_dtype=pl.Int32,
            )
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
        return _HIGHD_META_SCHEMA

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _HIGHD_TRACK_SCHEMA

    @staticmethod
    @override
    def location_id_select(meta_df: pl.DataFrame, path: Path) -> str:
        _ = meta_df
        return str(path)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_ACCELERATION


_META_SCHEMA: pl.Schema = pl.Schema({"trackId": pl.Int32, "class": pl.Utf8})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "trackId": pl.Int32,
    "xCenter": pl.Float64,
    "yCenter": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
})

_EXID_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "trackId": pl.Int32,
    "xCenter": pl.Float64,
    "yCenter": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
    "laneletId": pl.Int32,
})

_HIGHD_META_SCHEMA: pl.Schema = pl.Schema({"id": pl.Int32, "numLaneChanges": pl.Int8})

_HIGHD_TRACK_SCHEMA: pl.Schema = pl.Schema({
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
