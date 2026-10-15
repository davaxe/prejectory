from __future__ import annotations

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.datasets.opendd.loader import OpenDDLoader
from prejectory.datasets.opendd.maps import OpenDDMapProvider
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from prejectory.datasets.shared.presets import (
    benchmark_task,
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from prejectory.datasets.shared.resources import map_provider_resources_factory

_opendd_resources = map_provider_resources_factory(
    create=lambda _root, map_config: OpenDDMapProvider(map_config),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=210,
        sample_time=1 / 30,
        window_step=75,
        resample=linear_resample(up=1, down=3),
    ),
    screening=minimum_observations_screening(6),
    map=MapConfig(extraction=FullMapExtraction()),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="opendd",
    loader_cls=OpenDDLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=60)},
    default_task="benchmark",
    native_schema=OpenDDLoader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=456,
        max_frames=16916,
        enabled_by_default=True,
    ),
    map_provider_factory=_opendd_resources,
)
