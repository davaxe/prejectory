from __future__ import annotations

from prejectory.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from prejectory.core.categories import DatasetSplit
from prejectory.datasets.argoverse1.loader import Argoverse1Loader, Argoverse1LoaderOptions
from prejectory.datasets.argoverse1.maps import Argoverse1MapBuilder
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
from prejectory.datasets.shared.resources import named_shared_map_resources_factory

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

_open_argoverse1_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        ("MIA", root / "hd_maps" / "map_files" / "pruned_argoverse_MIA_10316_vector_map.xml"),
        ("PIT", root / "hd_maps" / "map_files" / "pruned_argoverse_PIT_10314_vector_map.xml"),
    ),
    build_map=lambda path, config: Argoverse1MapBuilder.from_xml_file(path).build(
        config.min_distance,
        config.interpolation_distance,
    ),
)


_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=50, sample_time=0.1),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    loader_options=Argoverse1LoaderOptions().model_dump(),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="argoverse1",
    loader_cls=Argoverse1Loader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=20)},
    default_task="benchmark",
    native_schema=Argoverse1Loader.native_trajectory_schema(),
    supported_native_splits=_NATIVE_SPLITS,
    loader_options_model=Argoverse1LoaderOptions,
    map_provider_factory=_open_argoverse1_resources,
    split_support=DatasetSplitSupport(scene=True),
    feature_support=DatasetFeatureSupport(map=True),
    temporal_support=temporal_support(
        source_unit="scenario",
        min_frames=20,
        max_frames=50,
        enabled_by_default=False,
    ),
)
