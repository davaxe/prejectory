from __future__ import annotations

import json
from itertools import starmap
from typing import TYPE_CHECKING, Any, ClassVar, override

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

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.core.scene import TrajectorySchema
    from prejectory.processing.maps import MapProvider
    from prejectory.processing.models import LoaderPlan


class NuScenesStyleLoaderOptions(LoaderOptionsModel):
    """Common loader options for nuScenes-style metadata datasets."""

    drop_status: list[str] = Field(default_factory=list)
    drop_full_category_regex: list[str] = Field(default_factory=lambda: ["object"])


class NuScenesStyleLoader(SceneLoader[str, NuScenesStyleLoaderOptions]):
    """Shared base for datasets that follow the nuScenes metadata layout."""

    metadata_dir_parts: ClassVar[tuple[tuple[str, ...], ...]] = ()
    native_splits: ClassVar[tuple[DatasetSplit, ...]] = ()
    source_identifier_field: ClassVar[str] = "name"
    source_data_field: ClassVar[str] = "token"
    bad_first_step_threshold_meters: ClassVar[float | None] = None

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, map_provider=map_provider)
        self._schemas: dict[str, pl.Schema | None] = _SCHEMAS
        self._sources: dict[DatasetSplit, list[DatasetSource[str]]] | None = None
        self._scene_cache: dict[str, pl.DataFrame] = {}

    def _build_sources_manifest(self) -> dict[DatasetSplit, list[DatasetSource[str]]]:
        sources: dict[DatasetSplit, list[DatasetSource[str]]] = {
            split: [] for split in self.native_splits
        }

        log_records = self._load_json_records("log")
        scene_records = self._load_json_records("scene")

        log_to_map = {
            str(row["token"]): str(row["location"]) if row.get("location") else None
            for row in log_records
            if "token" in row
        }

        for row in scene_records:
            if "token" not in row or "log_token" not in row:
                continue
            sources[self.split_from_scene_row(row)].append(
                DatasetSource(
                    identifier=self.source_identifier_from_scene_row(row),
                    payload=self.source_data_from_scene_row(row),
                    map_key=log_to_map.get(str(row["log_token"])),
                ),
            )
        return sources

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[str]]:
        if self._sources is None:
            self._sources = self._build_sources_manifest()
        yield from self._sources.get(split, [])

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if self._sources is None:
            self._sources = self._build_sources_manifest()
        return len(self._sources.get(split, []))

    @override
    def load_source(self, source: DatasetSource[str]) -> Iterable[LoadedSourceFrame]:
        self._ensure_dataset_loaded()
        scenes = self._scene_cache[source.payload].drop(["scene_token", "scene_name", "map"]).lazy()
        filters = [~pl.col("status").is_in(self.loader_options.drop_status)]
        filters.extend(
            ~pl.col("full_category").str.contains(regex)
            for regex in self.loader_options.drop_full_category_regex
        )
        yield LoadedSourceFrame(
            scenes.filter(*filters).drop(["status", "full_category", "full_status"]),
            ego_agent_id=0,
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    def _ensure_dataset_loaded(self) -> None:
        if self._scene_cache:
            return
        dfs = {name: self._load_table(name, schema) for name, schema in self._schemas.items()}
        timeline_lf = build_scene_timeline(dfs["sample"], dfs["scene"], dfs["log"])
        timeline_lf = self.prepare_timeline(timeline_lf, dfs["sample_data"], dfs["ego_pose"])
        ego_lf = extract_ego_tracks(timeline_lf, dfs["sample_data"], dfs["ego_pose"])
        agents_lf = extract_agent_tracks(
            timeline_lf,
            dfs["sample_annotation"],
            dfs["instance"],
            dfs["category"],
            dfs["attribute"],
            category_mapping=_FULL_CATEGORY_MAPPING,
            status_mapping=_STATUS_MAPPING,
        ).select(ego_lf.collect_schema().names())
        combined_df = (
            pl
            .concat([ego_lf, agents_lf])
            .sort(["scene_token", "frame", "id"])
            .collect(engine="streaming")
        )
        raw_partitions = combined_df.partition_by("scene_token", as_dict=True)
        self._scene_cache = {key[0]: df for key, df in raw_partitions.items()}

    def metadata_dirs(self) -> tuple[Path, ...]:
        """Return the dataset metadata roots resolved under `self.root`."""
        return tuple(starmap(self.root.joinpath, self.metadata_dir_parts))

    def prepare_timeline(
        self,
        timeline_lf: pl.LazyFrame,
        sample_data_lf: pl.LazyFrame,
        ego_pose_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Apply dataset-specific timeline cleanup before track extraction."""
        if self.bad_first_step_threshold_meters is None:
            return timeline_lf
        return drop_first_samples_with_large_ego_jump(
            timeline_lf,
            sample_data_lf,
            ego_pose_lf,
            threshold_meters=self.bad_first_step_threshold_meters,
        )

    def _load_json_records(self, name: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for data_dir in self.metadata_dirs():
            with (data_dir / f"{name}.json").open("r", encoding="utf-8") as f:
                records.extend(json.load(f))
        return records

    def _load_table(self, name: str, schema: pl.Schema | None = None) -> pl.LazyFrame:
        tables = [load_cached_table(name, data_dir, schema) for data_dir in self.metadata_dirs()]
        if len(tables) == 1:
            return tables[0]
        combined = pl.concat(tables, how="vertical")
        if schema is not None and "token" in schema:
            return combined.unique(subset=["token"], maintain_order=True)
        return combined

    @staticmethod
    def split_from_scene_row(row: dict[str, Any]) -> DatasetSplit:
        """Resolve the dataset split for one raw scene metadata row."""
        msg = f"{type(row).__name__} split resolution is not implemented."
        raise NotImplementedError(msg)

    @classmethod
    def source_identifier_from_scene_row(cls, row: dict[str, Any]) -> str:
        """Return the public DatasetSource identifier for one scene row."""
        return str(row[cls.source_identifier_field])

    @classmethod
    def source_data_from_scene_row(cls, row: dict[str, Any]) -> str:
        """Return the loader-internal DatasetSource payload for one scene row."""
        return str(row[cls.source_data_field])


def load_cached_table(name: str, base_dir: Path, schema: pl.Schema | None = None) -> pl.LazyFrame:
    """Load one JSON metadata table lazily with an optional schema."""
    return pl.read_json(base_dir / f"{name}.json", schema=schema).lazy()


def build_scene_timeline(
    sample_lf: pl.LazyFrame,
    scene_lf: pl.LazyFrame,
    log_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Join nuScenes-style `sample`, `scene`, and `log` source tables into a timeline."""
    return (
        sample_lf
        .join(scene_lf, left_on="scene_token", right_on="token")
        .join(log_lf, left_on="log_token", right_on="token")
        .select(
            pl.col("token").alias("sample_token"),
            "scene_token",
            pl.col("name").alias("scene_name"),
            pl.col("location").alias("map"),
            "timestamp",
        )
        .pipe(renumber_scene_timeline)
    )


def renumber_scene_timeline(timeline_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Assign dense zero-based frame indices within each scene."""
    return timeline_lf.with_columns(
        pl.col("timestamp").rank("dense").over("scene_token").sub(1).cast(pl.Int64).alias("frame"),
    )


def drop_first_samples_with_large_ego_jump(
    timeline_lf: pl.LazyFrame,
    sample_data_lf: pl.LazyFrame,
    ego_pose_lf: pl.LazyFrame,
    *,
    threshold_meters: float,
) -> pl.LazyFrame:
    """Drop scene-start rows whose next ego step exceeds a distance threshold."""
    bad_first_samples_lf = (
        timeline_lf
        .join(sample_data_lf.select(["sample_token", "ego_pose_token"]), on="sample_token")
        .group_by("sample_token", maintain_order=False)
        .first()
        .join(ego_pose_lf, left_on="ego_pose_token", right_on="token")
        .select(
            "scene_token",
            "sample_token",
            "timestamp",
            "frame",
            pl.col("translation").list.get(0).alias("x"),
            pl.col("translation").list.get(1).alias("y"),
        )
        .sort(["scene_token", "frame"])
        .with_columns(
            pl.col("x").shift(-1).over("scene_token").alias("next_x"),
            pl.col("y").shift(-1).over("scene_token").alias("next_y"),
        )
        .with_columns(
            ((pl.col("next_x") - pl.col("x")).pow(2) + (pl.col("next_y") - pl.col("y")).pow(2))
            .sqrt()
            .alias("next_displacement"),
        )
        .filter((pl.col("frame") == 0) & pl.col("next_displacement").gt(threshold_meters))
        .select("sample_token")
    )

    return timeline_lf.join(bad_first_samples_lf, on="sample_token", how="anti").pipe(
        renumber_scene_timeline,
    )


def extract_ego_tracks(
    timeline_lf: pl.LazyFrame,
    sample_data_lf: pl.LazyFrame,
    ego_pose_lf: pl.LazyFrame,
    ego_id: int = 0,
    ego_category: int = AgentCategory.CAR,
) -> pl.LazyFrame:
    """Extract ego positions aligned to the shared scene timeline."""
    return (
        timeline_lf
        .join(sample_data_lf.select(["sample_token", "ego_pose_token"]), on="sample_token")
        .group_by("sample_token", maintain_order=False)
        .first()
        .join(ego_pose_lf, left_on="ego_pose_token", right_on="token")
        .select(
            *("scene_token", "scene_name", "map", "frame"),
            pl.lit(ego_id, dtype=pl.Int32).alias("id"),
            pl.col("translation").list.get(0).alias("x"),
            pl.col("translation").list.get(1).alias("y"),
            pl.lit(ego_category, dtype=pl.Int32).alias("agent_category"),
            pl.lit("vehicle.ego.car").alias("full_category"),
            pl.lit("moving").alias("status"),
            pl.lit("moving").alias("full_status"),
        )
    )


def extract_agent_tracks(
    timeline_lf: pl.LazyFrame,
    annotation_lf: pl.LazyFrame,
    instance_lf: pl.LazyFrame,
    category_lf: pl.LazyFrame,
    attribute_lf: pl.LazyFrame,
    category_mapping: dict[str, int],
    status_mapping: dict[str, str],
    default_agent_category: int = AgentCategory.UNKNOWN,
    default_status: str = "unknown",
) -> pl.LazyFrame:
    """Extract annotated agent positions aligned to the shared scene timeline."""
    attr_lookup = attribute_lf.select(
        pl.col("token").alias("attr_token"),
        pl.col("name").alias("attr_name"),
    )
    cat_lookup = category_lf.select(
        pl.col("token").alias("cat_token"),
        pl.col("name").alias("cat_name"),
    )
    raw_agents = (
        annotation_lf
        .join(timeline_lf, on="sample_token")
        .join(instance_lf, left_on="instance_token", right_on="token")
        .join(cat_lookup, left_on="category_token", right_on="cat_token")
        .with_columns(pl.col("attribute_tokens").list.first().alias("first_attr_token"))
        .join(attr_lookup, left_on="first_attr_token", right_on="attr_token", how="left")
        .with_columns(
            pl.col("instance_token").rank("dense").over("scene_token").cast(pl.Int32).alias("id"),
        )
        .select(
            *("scene_token", "scene_name", "map", "frame", "id"),
            pl.col("translation").list.get(0).alias("x"),
            pl.col("translation").list.get(1).alias("y"),
            pl.col("cat_name").alias("full_category"),
            pl.col("attr_name").fill_null("unknown").alias("full_status"),
        )
    )

    return raw_agents.with_columns(
        pl
        .col("full_category")
        .replace_strict(category_mapping, default=None)
        .fill_null(default_agent_category)
        .cast(pl.Int32)
        .alias("agent_category"),
        pl
        .col("full_status")
        .replace_strict(status_mapping, default=default_status)
        .alias("status"),
    )


_SCHEMAS: dict[str, pl.Schema | None] = {
    "scene": pl.Schema({
        "token": pl.String,
        "log_token": pl.String,
        "nbr_samples": pl.Int32,
        "first_sample_token": pl.String,
        "last_sample_token": pl.String,
        "name": pl.String,
        "description": pl.String,
    }),
    "sample_annotation": pl.Schema({
        "token": pl.String,
        "sample_token": pl.String,
        "instance_token": pl.String,
        "visibility_token": pl.String,
        "attribute_tokens": pl.List(pl.String),
        "translation": pl.List(pl.Float64),
        "size": pl.List(pl.Float64),
        "rotation": pl.List(pl.Float64),
        "prev": pl.String,
        "next": pl.String,
        "num_lidar_pts": pl.Int32,
        "num_radar_pts": pl.Int32,
    }),
    "instance": pl.Schema({
        "token": pl.String,
        "category_token": pl.String,
        "nbr_annotations": pl.Int32,
        "first_annotation_token": pl.String,
        "last_annotation_token": pl.String,
    }),
    "category": pl.Schema({"token": pl.String, "name": pl.String, "description": pl.String}),
    "attribute": pl.Schema({"token": pl.String, "name": pl.String, "description": pl.String}),
    "ego_pose": pl.Schema({
        "token": pl.String,
        "timestamp": pl.Int64,
        "translation": pl.List(pl.Float64),
        "rotation": pl.List(pl.Float64),
    }),
    "log": pl.Schema({
        "token": pl.String,
        "logfile": pl.String,
        "vehicle": pl.String,
        "date_captured": pl.String,
        "location": pl.String,
    }),
    "sample": pl.Schema({
        "token": pl.String,
        "prev": pl.String,
        "next": pl.String,
        "scene_token": pl.String,
        "timestamp": pl.Int64,
    }),
    "sample_data": pl.Schema({
        "token": pl.String,
        "sample_token": pl.String,
        "ego_pose_token": pl.String,
        "calibrated_sensor_token": pl.String,
        "timestamp": pl.Int64,
        "fileformat": pl.String,
        "filename": pl.String,
        "prev": pl.String,
        "next": pl.String,
        "is_key_frame": pl.Boolean,
        "height": pl.Int32,
        "width": pl.Int32,
    }),
}

_FULL_CATEGORY_MAPPING: dict[str, int] = {
    "vehicle.car": AgentCategory.CAR,
    "vehicle.ego.car": AgentCategory.CAR,
    "vehicle.van": AgentCategory.VAN,
    "vehicle.construction": AgentCategory.VAN,
    "vehicle.bus.bendy": AgentCategory.BUS,
    "vehicle.bus.rigid": AgentCategory.BUS,
    "vehicle.truck": AgentCategory.TRUCK,
    "vehicle.trailer": AgentCategory.TRAILER,
    "vehicle.motorcycle": AgentCategory.MOTORCYCLE,
    "vehicle.bicycle": AgentCategory.BICYCLE,
    "vehicle.emergency.ambulance": AgentCategory.CAR,
    "vehicle.emergency.police": AgentCategory.CAR,
    "human.pedestrian.adult": AgentCategory.PEDESTRIAN,
    "human.pedestrian.child": AgentCategory.PEDESTRIAN,
    "human.pedestrian.construction_worker": AgentCategory.PEDESTRIAN,
    "human.pedestrian.police_officer": AgentCategory.PEDESTRIAN,
    "human.pedestrian.stroller": AgentCategory.PEDESTRIAN,
    "human.pedestrian.wheelchair": AgentCategory.PEDESTRIAN,
    "human.pedestrian": AgentCategory.PEDESTRIAN,
    "static_object.bicycle_rack": AgentCategory.STATIC_OBJECT,
    "movable_object.barrier": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.debris": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.pushable_pullable": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.trafficcone": AgentCategory.MOVEABLE_OBJECT,
    "animal": AgentCategory.ANIMAL,
}

_STATUS_MAPPING: dict[str, str] = {
    "vehicle.moving": "moving",
    "vehicle.stopped": "stopped",
    "vehicle.parked": "parked",
    "pedestrian.moving": "moving",
    "pedestrian.standing": "stopped",
    "pedestrian.sitting_lying_down": "stopped",
    "cycle.with_rider": "moving",
    "cycle.without_rider": "stopped",
}
