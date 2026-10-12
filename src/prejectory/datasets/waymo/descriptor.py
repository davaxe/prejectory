from __future__ import annotations

from prejectory.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from prejectory.core.categories import DatasetSplit
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from prejectory.datasets.shared.presets import (
    benchmark_task,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from prejectory.datasets.shared.resources import map_provider_resources_factory
from prejectory.datasets.waymo.loader import WaymoLoader
from prejectory.datasets.waymo.maps import WaymoEmbeddedMapProvider

_open_waymo_resources = map_provider_resources_factory(
    create=lambda _root, map_config: WaymoEmbeddedMapProvider(map_config),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=91, sample_time=0.1),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="waymo",
    loader_cls=WaymoLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=11)},
    default_task="benchmark",
    native_schema=WaymoLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
    temporal_support=temporal_support(
        source_unit="scenario",
        min_frames=11,
        max_frames=91,
        enabled_by_default=False,
        confidence="documented",
    ),
    map_provider_factory=_open_waymo_resources,
)
