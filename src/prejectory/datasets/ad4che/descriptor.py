from __future__ import annotations

from pathlib import Path

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.datasets.ad4che.loader import AD4CHELoader
from prejectory.datasets.ad4che.maps import AD4CHEMapProvider
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from prejectory.datasets.shared.presets import (
    benchmark_task,
    lane_change_sampling,
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from prejectory.datasets.shared.resources import map_provider_resources_factory

_open_ad4che_resources = map_provider_resources_factory(
    create=lambda root, map_config: AD4CHEMapProvider(
        root=Path(root) / "AD4CHE_Data_V1.0",
        config=map_config,
    ),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=211,
        sample_time=1 / 30,
        window_step=25,
        resample=linear_resample(up=1, down=3),
        lane_change=lane_change_sampling(required_lane_changes=5, negative_keep_every=3),
    ),
    map=MapConfig(extraction=FullMapExtraction(), interpolation_distance=8),
    screening=minimum_observations_screening(6),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="ad4che",
    loader_cls=AD4CHELoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=61)},
    default_task="benchmark",
    native_schema=AD4CHELoader.native_trajectory_schema(),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=1136,
        max_frames=9821,
        enabled_by_default=True,
    ),
    map_provider_factory=_open_ad4che_resources,
)
