from __future__ import annotations

from prejectory.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from prejectory.core.categories import DatasetSplit
from prejectory.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions
from prejectory.datasets.nuscenes.maps import NuScenesMapBuilder
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
from prejectory.datasets.shared.resources import named_shared_map_resources_factory

_open_nuscenes_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        (path.stem, path)
        for path in (root / "nuScenes-map-expansion-v1.3" / "expansion").glob("*.json")
    ),
    build_map=lambda path, config: NuScenesMapBuilder.from_json_file(path).build(
        config.min_distance,
        config.interpolation_distance,
    ),
)


_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=16,
        sample_time=0.5,
        window_step=1,
        resample=linear_resample(up=5),
    ),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="nuscenes",
    loader_cls=NuScenesLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=4)},
    default_task="benchmark",
    loader_options_model=NuScenesLoaderOptions,
    native_schema=NuScenesLoader.native_trajectory_schema(),
    map_provider_factory=_open_nuscenes_resources,
    feature_support=DatasetFeatureSupport(map=True),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    split_support=DatasetSplitSupport(scene=True, source=True),
    temporal_support=temporal_support(
        source_unit="scene",
        min_frames=32,
        max_frames=41,
        enabled_by_default=True,
    ),
)
