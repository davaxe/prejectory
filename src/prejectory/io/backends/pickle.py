"""Pickle writer backend.

This backend writes one pickled full-horizon `SceneRecord` per scene inside split
subdirectories such as `train` or `unsplit`. It is the simplest persisted
backend, requires no optional storage dependency, and is a practical default
when inspectability matters more than shard-based streaming.
"""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from prejectory.core.scene import get_trajectory_schema
from prejectory.io.base import DatasetWriter, split_directory_name
from prejectory.io.encoding import encode_scene_record

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.config.models import OutputConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.core.scene import Scene, TrajectorySchema
    from prejectory.io.records import PredictionBounds


class PickleWriter(DatasetWriter):
    """Write one pickled full-horizon `SceneRecord` per scene."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: OutputConfig,
        prediction_bounds: PredictionBounds | None = None,
        splits: Iterable[DatasetSplit] | None = None,
    ) -> None:
        self._config: OutputConfig = config
        self._trajectory_schema: TrajectorySchema = get_trajectory_schema(config.trajectory_schema)
        self._prediction_bounds: PredictionBounds | None = prediction_bounds

        self._dir_map: dict[DatasetSplit | None, Path] = {
            split: output_dir / split_directory_name(split) for split in splits or [None]
        }
        for sub_dir in self._dir_map.values():
            sub_dir.mkdir(parents=True, exist_ok=True)

    @override
    def write(self, scene: Scene) -> None:
        """Encode and persist one scene record."""
        output_dir = self._dir_map[scene.split_assignment]
        file_path = output_dir / f"{scene.scene_number:06d}.pkl"
        record = encode_scene_record(
            scene,
            dtype=np.float32 if self._config.precision == "float32" else np.float64,
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._trajectory_schema,
            prediction_bounds=self._prediction_bounds,
        )
        with file_path.open("wb", buffering=1024 * 1024) as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)
