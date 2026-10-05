# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
"""Registry-driven writer backend resolution.

The runtime keeps backend selection separate from scene encoding. A resolved
`RunPlan` chooses a storage backend name, and this registry maps that name to a
builder that creates a writer provider for one execution run.

The built-in registry entries are:

- `mds` for shard-based Mosaic Streaming output
- `pickle` for one pickled scene record per file
- `null` for dry-run style execution without persisted scene data
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, cast

from prejectory.io.base import StorageBackend, WorkerWriterProvider, WriterProvider

if TYPE_CHECKING:
    from pathlib import Path

    from prejectory.config.models import OutputConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.io.base import DatasetWriter, RecordTransform, SceneTransform
    from prejectory.io.records import PredictionBounds
    from prejectory.runtime.types import ExecutionPlan

logger = logging.getLogger(__name__)


def build_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    """Build the writer provider for one resolved processing plan."""
    match plan.storage_backend:
        case StorageBackend.MDS:
            return _build_mds_writer_provider(plan)
        case StorageBackend.PICKLE:
            return _build_pickle_writer_provider(plan)
        case StorageBackend.ZARR:
            return _build_zarr_writer_provider(plan)
        case StorageBackend.NULL:
            return WorkerWriterProvider(create_worker=_create_null_writer)


def _build_mds_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    from prejectory.io.backends.mds import MDSDatasetWriter  # ruff: ignore[import-outside-top-level]

    output_transform = plan.output_transform
    splits = _output_splits(plan)
    return WorkerWriterProvider(
        create_worker=functools.partial(
            _create_mds_writer,
            output_dir=plan.output_dir,
            config=plan.output_config,
            prediction_bounds=_prediction_bounds(plan),
            splits=splits,
            parallel=plan.parallel,
            record_transform=(
                None
                if output_transform is None
                else cast(
                    "RecordTransform[dict[str, Any]] | None",
                    output_transform.record_transform,
                )
            ),
            scene_transform=(
                None
                if output_transform is None
                else cast("SceneTransform[dict[str, Any]] | None", output_transform.scene_transform)
            ),
            mds_columns=None if output_transform is None else output_transform.mds_columns,
        ),
        finalize=functools.partial(
            MDSDatasetWriter.finish_dataset,
            output_dir=plan.output_dir,
            splits=splits,
            parallel=plan.parallel,
        ),
    )


def _create_mds_writer(
    worker_id: int,
    *,
    output_dir: Path,
    config: OutputConfig,
    prediction_bounds: PredictionBounds | None,
    splits: tuple[DatasetSplit, ...] | None,
    parallel: bool,
    record_transform: RecordTransform[dict[str, Any]] | None,
    scene_transform: SceneTransform[dict[str, Any]] | None,
    mds_columns: dict[str, str] | None,
) -> DatasetWriter:
    from prejectory.io.backends.mds import MDSDatasetWriter  # ruff: ignore[import-outside-top-level]

    return MDSDatasetWriter(
        output_dir=output_dir,
        config=config,
        prediction_bounds=prediction_bounds,
        splits=splits,
        parallel=parallel,
        parallel_group=worker_id,
        record_transform=record_transform,
        scene_transform=scene_transform,
        mds_columns=mds_columns,
    )


def _create_null_writer(worker_id: int) -> DatasetWriter:
    from prejectory.io.backends.null import NullWriter  # ruff: ignore[import-outside-top-level]

    _ = worker_id
    return NullWriter()


def _build_zarr_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    from prejectory.io.backends.zarr import ZarrDatasetWriter  # ruff: ignore[import-outside-top-level]

    splits = _output_splits(plan)
    return WorkerWriterProvider(
        create_worker=functools.partial(
            _create_zarr_writer,
            output_dir=plan.output_dir,
            config=plan.output_config,
            prediction_bounds=_prediction_bounds(plan),
            splits=splits,
        ),
        finalize=functools.partial(
            ZarrDatasetWriter.finish_dataset,
            output_dir=plan.output_dir,
            splits=splits,
        ),
    )


def _create_zarr_writer(
    worker_id: int,
    *,
    output_dir: Path,
    config: OutputConfig,
    prediction_bounds: PredictionBounds | None,
    splits: tuple[DatasetSplit, ...] | None,
) -> DatasetWriter:
    from prejectory.io.backends.zarr import ZarrDatasetWriter  # ruff: ignore[import-outside-top-level]

    return ZarrDatasetWriter(
        output_dir=output_dir,
        identifier=worker_id,
        config=config,
        prediction_bounds=prediction_bounds,
        splits=splits,
    )


def _build_pickle_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    output_transform = plan.output_transform
    return WorkerWriterProvider(
        create_worker=functools.partial(
            _create_pickle_writer,
            output_dir=plan.output_dir,
            config=plan.output_config,
            prediction_bounds=_prediction_bounds(plan),
            splits=_output_splits(plan),
            record_transform=None
            if output_transform is None
            else output_transform.record_transform,
            scene_transform=None if output_transform is None else output_transform.scene_transform,
        ),
    )


def _create_pickle_writer(
    worker_id: int,
    *,
    output_dir: Path,
    config: OutputConfig,
    prediction_bounds: PredictionBounds | None,
    splits: tuple[DatasetSplit, ...] | None,
    record_transform: RecordTransform[object] | None,
    scene_transform: SceneTransform[object] | None,
) -> DatasetWriter:
    from prejectory.io.backends.pickle import PickleWriter  # ruff: ignore[import-outside-top-level]

    return PickleWriter(
        output_dir=output_dir,
        identifier=worker_id,
        config=config,
        prediction_bounds=prediction_bounds,
        splits=splits,
        record_transform=record_transform,
        scene_transform=scene_transform,
    )


def _output_splits(plan: ExecutionPlan) -> tuple[DatasetSplit, ...] | None:
    return plan._assignment.output_splits(input_native_splits=plan._loader.read.native_splits)


def _prediction_bounds(plan: ExecutionPlan) -> PredictionBounds | None:
    return plan.effective_prediction_bounds
