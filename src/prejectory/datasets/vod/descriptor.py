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
from prejectory.datasets.shared.resources import single_shared_map_resource_factory
from prejectory.datasets.vod.loader import VodLoader, VodLoaderOptions
from prejectory.datasets.vod.maps import VodMapBuilder

_open_vod_resources = single_shared_map_resource_factory(
    map_path=lambda root: root / "maps" / "expansion" / "delft.json",
    build_map=lambda path, config: VodMapBuilder.from_json_file(path).build(
        config.min_distance,
        config.interpolation_distance,
    ),
)


_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=35, sample_time=0.1, window_step=5),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="vod",
    loader_cls=VodLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=5)},
    default_task="benchmark",
    loader_options_model=VodLoaderOptions,
    native_schema=VodLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    map_provider_factory=_open_vod_resources,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=False, time_block=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=36,
        max_frames=784,
        enabled_by_default=True,
    ),
)
