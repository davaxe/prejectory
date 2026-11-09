from __future__ import annotations

from typing import ClassVar, override

from prejectory.core.categories import DatasetSplit
from prejectory.datasets.shared.nuscenes_style_loader import (
    NuScenesStyleLoader,
    NuScenesStyleLoaderOptions,
)
from prejectory.datasets.vod.splits import get_split

VodLoaderOptions = NuScenesStyleLoaderOptions


class VodLoader(NuScenesStyleLoader):
    """Loader for the View-of-Delft dataset."""

    metadata_dir_parts: ClassVar[tuple[tuple[str, ...], ...]] = (("v1.0-trainval",), ("v1.0-test",))
    native_splits: ClassVar[tuple[DatasetSplit, ...]] = (
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    )
    source_identifier_field: ClassVar[str] = "token"
    bad_first_step_threshold_meters: ClassVar[float | None] = 15.0

    @staticmethod
    @override
    def split_from_scene_row(row: dict[str, object]) -> DatasetSplit:
        """Resolve the native split for one VoD scene row."""
        return get_split(str(row["token"]))
