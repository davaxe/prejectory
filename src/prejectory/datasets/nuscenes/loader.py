from __future__ import annotations

from typing import ClassVar, override

from prejectory.core.categories import DatasetSplit
from prejectory.datasets.nuscenes.splits import get_split
from prejectory.datasets.shared.nuscenes_style_loader import (
    NuScenesStyleLoader,
    NuScenesStyleLoaderOptions,
)

NuScenesLoaderOptions = NuScenesStyleLoaderOptions


class NuScenesLoader(NuScenesStyleLoader):
    """Loader for nuScenes trajectories."""

    metadata_dir_parts: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("v1.0-trainval_meta", "v1.0-trainval"),
    )
    native_splits: ClassVar[tuple[DatasetSplit, ...]] = (DatasetSplit.TRAIN, DatasetSplit.VAL)

    @staticmethod
    @override
    def split_from_scene_row(row: dict[str, object]) -> DatasetSplit:
        """Resolve the native split for one nuScenes scene row."""
        return get_split(str(row["name"]))
