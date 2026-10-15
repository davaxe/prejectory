from __future__ import annotations

import polars as pl

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.datasets.ngsim.loader import NGSimLoader
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from prejectory.datasets.shared.highway_builder import HighwayLaneMapBuilder, LaneDescription
from prejectory.datasets.shared.presets import (
    benchmark_task,
    lane_change_sampling,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from prejectory.datasets.shared.resources import (
    MapProviderFactory,
    single_shared_map_resource_factory,
)

_SELECT_EXPR = (
    pl.col("Vehicle_ID").alias("id"),
    pl.col("Local_X").alias("x").mul(0.3048),
    pl.col("Local_Y").alias("y").mul(0.3048),
    pl.col("Lane_ID").alias("lane_id"),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=70,
        sample_time=0.1,
        window_step=25,
        lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
    ),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=FullMapExtraction()),
)


def ngsim_resources(
    lane_description: LaneDescription | None = None,
    bin_size: float = 10,
    smoothing: float | None = 3.0,
    *,
    include_outer_borders: bool = True,
) -> MapProviderFactory:
    """Create a map-provider factory for the NGSIM dataset."""
    return single_shared_map_resource_factory(
        map_path=lambda root: root,
        build_map=lambda path, config: HighwayLaneMapBuilder.from_csv(
            *path.rglob("trajectories*.csv"),
            select_expr=_SELECT_EXPR,
            bin_size=bin_size,
            lane_description=lane_description,
            include_outer_borders=include_outer_borders,
            smoothing=smoothing,
        ).build(config.min_distance, config.interpolation_distance),
    )


DATASET_DESCRIPTORS = {
    "i80": DatasetDescriptor(
        name="i80",
        loader_cls=NGSimLoader,
        default_config=_DEFAULT_CONFIG,
        tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=20)},
        default_task="benchmark",
        native_schema=NGSimLoader.native_trajectory_schema(),
        map_provider_factory=ngsim_resources(
            LaneDescription(ids=list(range(1, 8)), direction=[True] * 7),
        ),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
        split_support=DatasetSplitSupport(scene=True, time_block=True),
        temporal_support=temporal_support(
            source_unit="recording",
            min_frames=7887,
            max_frames=8738,
            enabled_by_default=True,
        ),
    ),
    "us101": DatasetDescriptor(
        name="us101",
        loader_cls=NGSimLoader,
        default_config=_DEFAULT_CONFIG,
        tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=20)},
        default_task="benchmark",
        native_schema=NGSimLoader.native_trajectory_schema(),
        map_provider_factory=ngsim_resources(
            LaneDescription(ids=list(range(1, 9)), direction=[True] * 8),
        ),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
        split_support=DatasetSplitSupport(scene=True, time_block=True),
        temporal_support=temporal_support(
            source_unit="recording",
            min_frames=7180,
            max_frames=8899,
            enabled_by_default=True,
        ),
    ),
}
