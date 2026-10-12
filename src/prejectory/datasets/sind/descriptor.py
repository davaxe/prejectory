from __future__ import annotations

from prejectory.config.models import DatasetConfig, FullMapExtraction, MapConfig
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from prejectory.datasets.shared.osm_builder import OSMMapBuilder
from prejectory.datasets.shared.presets import (
    benchmark_task,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from prejectory.datasets.shared.resources import named_shared_map_resources_factory
from prejectory.datasets.sind.loader import SindLoader

_open_sind_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        ("Changchun", root / "Changchun" / "Changchun_Pudong.osm"),
        ("Xi'an", root / "Xi'an" / "Xi'an_Shanglin.osm"),
        ("Chongqing", root / "Chongqing" / "NR_ll2.osm"),
        ("Tianjin", root / "Tianjin" / "map_relink_law_save.osm"),
    ),
    build_map=lambda path, config: OSMMapBuilder(
        path,
        force_zone_from_origin=(0, 0),
        local_origin_latlon=(0, 0),
    ).build(config.min_distance, config.interpolation_distance),
)


_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(horizon_frames=70, sample_time=0.1, window_step=25),
    screening=minimum_observations_screening(2),
    map=MapConfig(extraction=FullMapExtraction()),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="sind",
    loader_cls=SindLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=20)},
    default_task="benchmark",
    native_schema=SindLoader.native_trajectory_schema(),
    map_provider_factory=_open_sind_resources,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=4715,
        max_frames=16023,
        enabled_by_default=True,
    ),
)
