from __future__ import annotations

from prejectory.config.models import DatasetConfig
from prejectory.datasets.apolloscape.loader import ApolloScapeLoader
from prejectory.datasets.registry import DatasetDescriptor, DatasetSplitSupport
from prejectory.datasets.shared.presets import (
    benchmark_task,
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=10,
        sample_time=0.5,
        window_step=1,
        resample=linear_resample(up=5),
    ),
    screening=minimum_observations_screening(2),
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="apolloscape",
    loader_cls=ApolloScapeLoader,
    default_config=_DEFAULT_CONFIG,
    tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=4)},
    default_task="benchmark",
    native_schema=ApolloScapeLoader.native_trajectory_schema(),
    split_support=DatasetSplitSupport(scene=True, source=True),
    temporal_support=temporal_support(
        source_unit="recording",
        min_frames=37,
        max_frames=119,
        enabled_by_default=True,
    ),
)
