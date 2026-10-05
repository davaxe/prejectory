"""Mosaic Streaming writer backend.

This backend writes processed scenes as MosaicML Streaming shards. It is the
shard-based option intended for training pipelines that benefit from streaming,
compression, and the upstream `StreamingDataset` reader surface.

Notes
-----
- requires the optional `prejectory[mds]` extra
- writes data into split subdirectories such as `train` or `unsplit`
- when running in parallel, each worker writes to its own temporary subfolder
  and the backend merges the shard indexes at the end of the run
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

from prejectory.core.errors import ConfigurationError
from prejectory.core.optional import raise_missing_optional_dependency
from prejectory.core.scene import get_trajectory_schema
from prejectory.io.base import (
    DatasetWriter,
    RecordTransform,
    SceneTransform,
    split_directory_name,
    validate_transform_choice,
)
from prejectory.io.encoding import encode_scene_record
from prejectory.io.encoding.mds import encode_mds_row
from prejectory.io.encoding.mds import mds_columns as default_mds_columns

try:
    from streaming import MDSWriter
    from streaming.base.util import merge_index
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS scene writer", extra="mds")


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from prejectory.config.models import OutputConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.core.scene import Scene, TrajectorySchema
    from prejectory.io.records import PredictionBounds


class MDSDatasetWriter(DatasetWriter):
    """Write processed scene records to MosaicML Streaming shards.

    By default each shard row contains the standard Prejectory `SceneRecord`
    payload encoded as MDS columns. Advanced callers may provide
    `record_transform` plus `mds_columns` to write custom MDS-compatible
    dictionaries derived from the encoded record, or `scene_transform` plus
    `mds_columns` to bypass record encoding and derive rows directly from
    the runtime `Scene`.

    Parameters
    ----------
    output_dir : Path
        The base output directory for the dataset. The writer will create split
        subdirectories such as `train` or `unsplit` as needed.
    config : OutputConfig
        The output configuration for the dataset, which controls encoding and
        MDS writer options.
    splits : Iterable[DatasetSplit], optional
        The dataset splits to write, e.g. `train` or `unsplit`. If not provided,
        the writer will write to the "unsplit" subdirectory by default.
    parallel : bool
        Whether the writer will be used in a parallel execution context. If True,
        the writer will write to worker-local temporary subdirectories and merge
        the shard indexes at the end of the run.
    parallel_group : int or str, optional
        An optional identifier for the parallel worker group. If not provided,
        the writer will use the current process name as the group identifier for
        parallel execution. This is only relevant if `parallel` is True.
    record_transform : RecordTransform[dict[str, Any]], optional
        A callable that transforms the encoded `SceneRecord` into a dictionary
        of MDS column values to be written as a row. This is the preferred
        customization hook for users who want to write custom MDS-compatible
        rows.
    scene_transform : SceneTransform[dict[str, Any]], optional
        A callable that transforms the runtime `Scene` directly into a
        dictionary of MDS column values to be written as a row.
    mds_columns : dict[str, str], optional
        A mapping from row field names to MDS column names. This is required
        if either `record_transform` or `scene_transform` is provided, and is
        ignored otherwise since the writer will use the default Prejectory MDS
        encoding scheme. See [MosaicML docs](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
        for more details on the expected column layout.

    """

    def __init__(
        self,
        output_dir: Path,
        *,
        config: OutputConfig,
        prediction_bounds: PredictionBounds | None = None,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        parallel_group: int | str | None = None,
        record_transform: RecordTransform[dict[str, Any]] | None = None,
        scene_transform: SceneTransform[dict[str, Any]] | None = None,
        mds_columns: dict[str, str] | None = None,
    ) -> None:
        validate_transform_choice(
            record_transform=record_transform,
            scene_transform=scene_transform,
        )
        if (record_transform is not None or scene_transform is not None) and mds_columns is None:
            msg = "Custom MDS transforms require `mds_columns`."
            raise ValueError(msg)
        self._base_output_dir: Path = Path(output_dir)
        self._config: OutputConfig = config
        self._trajectory_schema: TrajectorySchema = get_trajectory_schema(config.trajectory_schema)
        self._prediction_bounds: PredictionBounds | None = prediction_bounds
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._parallel_group: str | int | None = parallel_group
        self._record_transform: RecordTransform[dict[str, Any]] | None = record_transform
        self._scene_transform: SceneTransform[dict[str, Any]] | None = scene_transform
        self._mds_columns: dict[str, str] = (
            mds_columns if mds_columns is not None else default_mds_columns(config.precision)
        )
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None

    def _init_writers(self) -> dict[DatasetSplit | None, MDSWriter]:
        writers: dict[DatasetSplit | None, MDSWriter] = {}
        config = self._config.mds
        group_name = (
            self._parallel_group
            if self._parallel_group not in {None, ""}
            else mp.current_process().name
        )
        for split in self._splits or (None,):
            split_dir = self._base_output_dir / split_directory_name(split)
            final_dir = split_dir / str(group_name) if self._parallel else split_dir
            if sys.platform == "win32":
                # MDSWriter cannot handle the C:\ part of a windows path. Below is
                # a workaround that might mess up cloud based paths.
                path_str = "/" + final_dir.relative_to(final_dir.anchor).as_posix()
            else:
                path_str = final_dir.as_posix()
            writers[split] = MDSWriter(
                out=path_str,
                columns=self._mds_columns,
                compression=config.compression,
                hashes=(list(config.hashes) if config.hashes is not None else None),
                size_limit=config.size_limit,
                exist_ok=config.exist_ok,
            )
        return writers

    @override
    def write(self, scene: Scene) -> None:
        """Encode and write one scene to the split-specific shard."""
        if self._writers is None:
            self._writers = self._init_writers()

        split: DatasetSplit | None = scene.split_assignment
        if split not in self._writers:
            msg = (
                f"Scene {scene.scene_number} belongs to split {split}, "
                "but no writer is configured for this split."
            )
            raise ConfigurationError(msg)

        effective_scene = scene.with_split_assignment(split) if split is not None else scene
        self._writers[split].write(self._make_row(effective_scene))

    def _make_row(self, scene: Scene) -> dict[str, Any]:
        if self._scene_transform is not None:
            return dict(self._scene_transform(scene))

        encoded_scene = encode_scene_record(
            scene,
            dtype=np.float32 if self._config.precision == "float32" else np.float64,
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._trajectory_schema,
            prediction_bounds=self._prediction_bounds,
        )
        if self._record_transform is not None:
            return dict(self._record_transform(encoded_scene))
        return dict(encode_mds_row(encoded_scene))

    @override
    def finish_local(self) -> None:
        """Finish the local shard writers for the current worker."""
        if self._writers is None:
            return
        for writer in self._writers.values():
            writer.finish()
        self._writers = None

    def finish_final(self) -> None:
        """Finalize dataset-wide MDS output after all workers finish."""
        self.finish_dataset(
            output_dir=self._base_output_dir,
            splits=self._splits,
            parallel=self._parallel,
        )

    @staticmethod
    def finish_dataset(
        *,
        output_dir: Path,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
    ) -> None:
        """Finalize dataset-wide MDS output after all workers finish."""
        if not parallel:
            return
        for split in tuple(splits or ()) or (None,):
            with _suppress_output():
                merge_index(str(output_dir / split_directory_name(split)), keep_local=True)


@contextmanager
def _suppress_output() -> Generator[None]:
    logger = logging.getLogger("streaming.base.storage.upload")
    old_level = logger.level
    old_disabled = logger.disabled
    try:
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        with (
            Path(os.devnull).open("w", encoding="utf-8") as devnull,
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            yield
    finally:
        logger.setLevel(old_level)
        logger.disabled = old_disabled
