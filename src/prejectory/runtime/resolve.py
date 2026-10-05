"""Runtime planning helpers for request-to-plan bootstrap."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import prejectory.core.errors as prejectory_exceptions
from prejectory.config.base import Clear
from prejectory.config.models import (
    AssignConfig,
    DatasetConfig,
    NoAssign,
    PreserveNativeAssign,
    ReadConfig,
    ReadNative,
    SceneAssign,
    ShuffledTimeBlockAssign,
    SourceAssign,
    TimeBlockAssign,
)
from prejectory.config.parse import ProjectConfig, parse_config
from prejectory.io.base import StorageBackend
from prejectory.io.records import PredictionBounds
from prejectory.processing.models import SplitAssignmentPlan
from prejectory.runtime.types import (
    ExecutionPlan,
    build_loader_plan,
    resolve_effective_scene_window,
)

if TYPE_CHECKING:
    from prejectory.datasets.registry import DatasetDescriptor
    from prejectory.runtime.types import ExecutionRequest, OutputTransform


logger = logging.getLogger(__name__)


def build_execution_plan(
    *,
    descriptor: DatasetDescriptor,
    request: ExecutionRequest,
) -> ExecutionPlan:
    """Build a full runtime plan from a public execution request."""
    _validate_input_path(request)
    _validate_output_path(request)
    include_map = request.include_map
    storage_backend = _resolve_storage_backend(request.storage_backend)
    validate_output_options(storage_backend, request.output_transform)
    resolved_config, selected_task = _resolve_dataset_config(
        descriptor=descriptor,
        request=request,
    )
    _validate_read_support(descriptor, resolved_config.read)
    _validate_assignment_support(descriptor, resolved_config.assign)
    _validate_feature_support(descriptor, resolved_config)
    diagnostics = _validate_temporal_support(descriptor, resolved_config)
    loader_request = build_loader_plan(
        descriptor=descriptor,
        resolved_config=resolved_config,
        include_map=include_map,
    )
    assignment_request = SplitAssignmentPlan.from_config(resolved_config.assign, seed=request.seed)
    effective_horizon_frames, effective_prediction_bounds, effective_sample_time = (
        resolve_effective_scene_window(resolved_config)
    )
    logger.debug(
        "Built execution plan",
        extra={
            "dataset": descriptor.name,
            "storage_backend": storage_backend,
            "parallel": resolved_config.runtime.jobs > 1,
            "include_map": loader_request.map is not None,
        },
    )
    return ExecutionPlan(
        _descriptor=descriptor,
        diagnostics=diagnostics,
        selected_task=selected_task,
        input_dir=request.input_dir.resolve(),
        output_dir=request.output_dir.resolve(),
        storage_backend=StorageBackend(storage_backend),
        _loader=loader_request,
        _assignment=assignment_request,
        effective_horizon_frames=effective_horizon_frames,
        effective_prediction_bounds=None
        if effective_prediction_bounds is None
        else PredictionBounds(*effective_prediction_bounds),
        effective_sample_time=effective_sample_time,
        output_transform=(
            None
            if request.output_transform is None
            else replace(
                request.output_transform,
                mds_columns=None
                if request.output_transform.mds_columns is None
                else dict(request.output_transform.mds_columns),
            )
        ),
        limit=request.limit,
        seed=request.seed,
        overwrite=request.overwrite,
        config=resolved_config,
    )


def _validate_read_support(descriptor: DatasetDescriptor, config: ReadConfig | None) -> None:
    if not isinstance(config, ReadNative):
        return
    supported = descriptor.supported_native_splits
    if supported and (config.splits is None or set(config.splits).issubset(supported)):
        return
    msg = f"Dataset {descriptor.name} does not support the requested read configuration."
    raise prejectory_exceptions.ConfigurationError(msg)


def _validate_assignment_support(
    descriptor: DatasetDescriptor,
    config: AssignConfig | None,
) -> None:
    if config is None:
        return
    support = descriptor.split_support
    match config:
        case NoAssign():
            supported = True
        case PreserveNativeAssign():
            supported = bool(descriptor.supported_native_splits)
        case TimeBlockAssign() | ShuffledTimeBlockAssign():
            supported = support.time_block
        case SceneAssign():
            supported = support.scene
        case SourceAssign():
            supported = support.source
    if not supported:
        msg = f"Dataset {descriptor.name} does not support the requested assignment configuration."
        raise prejectory_exceptions.ConfigurationError(msg)


def _validate_feature_support(descriptor: DatasetDescriptor, config: DatasetConfig) -> None:
    if config.scenes.lane_change is None:
        return
    if config.scenes.window is None:
        msg = "Lane-change sampling requires window sampling to be enabled."
        raise prejectory_exceptions.ConfigurationError(msg)
    if not descriptor.feature_support.lane_change_sampling:
        msg = f"Dataset {descriptor.name} does not support lane-change sampling."
        raise prejectory_exceptions.ConfigurationError(msg)


def _validate_temporal_support(
    descriptor: DatasetDescriptor, config: DatasetConfig
) -> tuple[str, ...]:
    support = descriptor.temporal_support
    window = config.scenes.window
    if support is None or window is None:
        return ()

    windowing = support.windowing
    if window.policy not in windowing.supported_policies:
        msg = (
            f"Dataset {descriptor.name} does not support window policy '{window.policy}'. "
            f"Supported policies: {', '.join(windowing.supported_policies)}."
        )
        raise prejectory_exceptions.ConfigurationError(msg)
    if windowing.validation == "off":
        return ()

    requested_frames = config.scenes.horizon_frames
    max_frames = (
        windowing.max_window_frames
        if windowing.max_window_frames is not None
        else support.source_frame_bounds.max_frames
    )
    if max_frames is None or requested_frames <= max_frames:
        return ()

    msg = (
        f"Dataset {descriptor.name} supports windows up to {max_frames} source frames, "
        f"but the resolved scene window requests {requested_frames} frames."
    )
    if windowing.validation == "warn":
        logger.warning(msg)
        return (msg,)
    raise prejectory_exceptions.ConfigurationError(msg)


def _validate_input_path(request: ExecutionRequest) -> None:
    if not request.input_dir.exists() and request.input_dir_exists:
        msg = f"Input directory {request.input_dir} does not exist."
        raise FileNotFoundError(msg)
    if request.input_dir_exists and not request.input_dir.is_dir():
        msg = f"Input directory {request.input_dir} is not a directory."
        raise NotADirectoryError(msg)


def _validate_output_path(request: ExecutionRequest) -> None:
    if request.output_dir.exists() and not request.output_dir.is_dir():
        msg = f"Output directory {request.output_dir} is not a directory."
        raise NotADirectoryError(msg)
    input_path = request.input_dir.resolve()
    output_path = request.output_dir.resolve()
    if (
        input_path == output_path
        or input_path.is_relative_to(output_path)
        or output_path.is_relative_to(input_path)
    ):
        msg = "Input and output directories must not overlap."
        raise prejectory_exceptions.ConfigurationError(msg)


def _resolve_storage_backend(storage_backend: StorageBackend | str) -> StorageBackend:
    try:
        resolved: StorageBackend = StorageBackend(storage_backend)
    except ValueError as exc:
        raise prejectory_exceptions.UnsupportedStorageBackendError(
            storage_backend,
            tuple(sb.value for sb in StorageBackend),
        ) from exc
    return resolved


def _resolve_dataset_config(
    *,
    descriptor: DatasetDescriptor,
    request: ExecutionRequest,
) -> tuple[DatasetConfig, str | None]:
    authored = request.config
    project = (
        authored
        if isinstance(authored, ProjectConfig)
        else (parse_config(authored) if authored is not None else ProjectConfig())
    )
    config = project.resolve_dataset_config(descriptor)
    selected_task = descriptor.default_task
    project_task = project.task_selection_for(descriptor.name)
    if isinstance(project_task, str):
        selected_task = project_task
    elif project_task is not None:
        selected_task = None

    patch = request.overrides
    if patch.task is not None:
        selected_task = None
    if "task" in request.model_fields_set:
        task = request.task
        if isinstance(task, str):
            selected_task = task
            if task not in descriptor.tasks:
                available = ", ".join(sorted(descriptor.tasks)) or "none"
                msg = (
                    f"Unknown task '{task}' for dataset {descriptor.name}. "
                    f"Available tasks: {available}."
                )
                raise prejectory_exceptions.ConfigurationError(
                    msg,
                )
            task = descriptor.tasks[task]
        else:
            selected_task = None
        patch = patch.model_copy(update={"task": Clear() if task is None else task})
    config = patch.merge_into(config).model_copy(deep=True)
    logger.debug("Resolved dataset config", extra={"dataset": descriptor.name})
    return config, selected_task


def validate_output_options(
    backend: StorageBackend,
    transform: OutputTransform[object] | None,
) -> None:
    """Check backend dependencies and transform contracts before writing anything."""
    if backend == StorageBackend.MDS:
        if transform is not None and not transform.mds_columns:
            msg = "Custom MDS transforms require `mds_columns`."
            raise prejectory_exceptions.ConfigurationError(msg)
        from prejectory.io.backends.mds import MDSDatasetWriter  # ruff: ignore[import-outside-top-level]

        _ = MDSDatasetWriter
    elif backend == StorageBackend.ZARR:
        if transform is not None:
            msg = "The Zarr backend only supports the canonical SceneRecord payload."
            raise prejectory_exceptions.ConfigurationError(msg)
        from prejectory.io.backends.zarr import ZarrDatasetWriter  # ruff: ignore[import-outside-top-level]

        _ = ZarrDatasetWriter
    elif transform is not None and transform.mds_columns is not None:
        msg = "mds_columns is only supported by the MDS backend."
        raise prejectory_exceptions.ConfigurationError(msg)
    if backend == StorageBackend.NULL and transform is not None:
        msg = "The null backend does not support output transforms."
        raise prejectory_exceptions.ConfigurationError(msg)
