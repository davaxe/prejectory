from __future__ import annotations

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.core.categories import DatasetSplit
from prejectory.datasets.argoverse2.loader import Argoverse2Loader, Argoverse2LoaderOptions
from prejectory.datasets.argoverse2.maps import Argoverse2MapProvider
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

_open_argoverse2_resources = map_provider_resources_factory(
    create=lambda _root, map_config: Argoverse2MapProvider(map_config),
)

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=110, sample_time=0.1),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=FullMapExtraction()),
    loader_options=Argoverse2LoaderOptions().model_dump(),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="argoverse2",
    loader_cls=Argoverse2Loader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=50)},
    default_task="benchmark",
    native_schema=Argoverse2Loader.native_trajectory_schema(),
    supported_native_splits=_NATIVE_SPLITS,
    loader_options_model=Argoverse2LoaderOptions,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
    temporal_support=temporal_support(
        source_unit="scenario",
        min_frames=50,
        max_frames=110,
        enabled_by_default=False,
    ),
    map_provider_factory=_open_argoverse2_resources,
)
