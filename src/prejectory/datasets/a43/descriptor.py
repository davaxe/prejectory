from __future__ import annotations

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.datasets.a43.loader import A43Loader
from prejectory.datasets.a43.maps import A43MapProvider
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

_open_a43_resources = map_provider_resources_factory(
    create=lambda _root, map_config: A43MapProvider(map_config),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=70, sample_time=0.1, window_step=25),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=FullMapExtraction()),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="a43",
    loader_cls=A43Loader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=20)},
    default_task="benchmark",
    native_schema=A43Loader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, time_block=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=52123,
        max_frames=52123,
        enabled_by_default=True,
    ),
    map_provider_factory=_open_a43_resources,
)
