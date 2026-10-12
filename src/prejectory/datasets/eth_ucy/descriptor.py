from __future__ import annotations

from prejectory.config.models import DatasetConfig
from prejectory.core.categories import DatasetSplit
from prejectory.datasets.eth_ucy.loader import EthUcyLoader
from prejectory.datasets.registry import DatasetDescriptor, DatasetSplitSupport
from prejectory.datasets.shared.presets import (
    benchmark_task,
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=20,
        sample_time=0.4,
        window_step=1,
        resample=linear_resample(up=4),
    ),
    screening=minimum_observations_screening(2),
)


def _descriptor(name: str) -> DatasetDescriptor:
    bounds = {
        "eth": (89, 934),
        "hotel": (89, 1168),
        "univ": (148, 934),
        "zara1": (89, 934),
        "zara2": (89, 1052),
    }
    min_frames, max_frames = bounds[name]
    return DatasetDescriptor(
        name=name,
        loader_cls=EthUcyLoader,
        default_config=_DEFAULT_CONFIG,
        tasks={"benchmark": benchmark_task(_DEFAULT_CONFIG.scenes, prediction_origin=8)},
        default_task="benchmark",
        native_schema=EthUcyLoader.native_trajectory_schema(),
        supported_native_splits=_NATIVE_SPLITS,
        split_support=DatasetSplitSupport(scene=True, source=True),
        temporal_support=temporal_support(
            source_unit="recording",
            min_frames=min_frames,
            max_frames=max_frames,
            enabled_by_default=True,
        ),
    )


DATASET_DESCRIPTORS = {
    name: _descriptor(name) for name in ("eth", "hotel", "univ", "zara1", "zara2")
}
