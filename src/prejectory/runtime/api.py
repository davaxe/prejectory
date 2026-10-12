# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
"""Public runtime entry points."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from prejectory.datasets.registry import get_dataset
from prejectory.io.backends.provider import build_writer_provider
from prejectory.io.base import split_directory_name
from prejectory.io.manifest import write_manifest
from prejectory.runtime.executor import open_executor
from prejectory.runtime.observer import observe_execution
from prejectory.runtime.resolve import build_execution_plan, validate_backend_dependencies
from prejectory.runtime.state import split_key
from prejectory.runtime.types import ExecutionPlan, ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from prejectory.runtime.state import Progress

logger = logging.getLogger(__name__)


def resolve_request(request: ExecutionRequest) -> ExecutionPlan:
    """Resolve defaults, configuration and overrides without writing output."""
    descriptor = get_dataset(request.dataset)
    return build_execution_plan(descriptor=descriptor, request=request)


def execute_request(
    request: ExecutionRequest,
    *,
    show_progress: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
) -> ExecutionResult:
    """Resolve and execute a request; see :func:`execute_plan` for progress semantics."""
    return execute_plan(
        resolve_request(request), show_progress=show_progress, on_progress=on_progress
    )


def execute_plan(
    plan: ExecutionPlan,
    *,
    show_progress: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
) -> ExecutionResult:
    """Execute the inspected plan, publishing output only on success.

    Python runs are quiet by default; `show_progress` selects a Rich display.
    `on_progress` receives snapshots on one background thread, including a
    final snapshot. It must be thread-safe and should return promptly. Callback
    errors propagate after execution stops and prevent publishing the output.
    Callbacks report processing progress, not successful publication.

    With `overwrite=True`, the old output is retained until the replacement
    is complete. Publication uses same-filesystem renames with rollback if the
    final rename fails; it is not a crash-atomic directory exchange.
    """
    validate_backend_dependencies(plan.storage_backend)
    _check_output_directory(plan.output_dir, overwrite=plan.overwrite)
    start = time.perf_counter()
    plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{plan.output_dir.name}-", dir=plan.output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "output"
        staged_plan = replace(plan, output_dir=staging)
        writer_provider = build_writer_provider(staged_plan)
        with open_executor(staged_plan) as executor:
            if show_progress:
                from prejectory.runtime.progress import execute_with_rich_progress  # ruff: ignore[import-outside-top-level]

                progress = execute_with_rich_progress(
                    executor,
                    lambda: executor.execute(writer_provider),
                    on_progress=on_progress,
                )
            else:
                progress = observe_execution(
                    executor, lambda: executor.execute(writer_provider), on_progress
                )
            cleanup = executor.cleanup_summary()
        output_splits = plan._assignment.output_splits(
            input_native_splits=plan._loader.read.native_splits
        )
        split_counts = {
            split_directory_name(split): progress.stats.split_counts[split_key(split)]
            for split in output_splits or (None,)
        }
        # Empty MDS exports may never initialize a writer.
        for name in split_counts:
            (staging / name).mkdir(parents=True, exist_ok=True)
        write_manifest(staging, replace(plan.manifest(), split_counts=split_counts))
        _check_output_directory(plan.output_dir, overwrite=plan.overwrite)
        previous = Path(temporary) / "previous"
        if plan.output_dir.exists():
            _ = plan.output_dir.rename(previous)
        try:
            _ = staging.rename(plan.output_dir)
        except OSError:
            if previous.exists():
                _ = previous.rename(plan.output_dir)
            raise

    logger.info(
        "Finished execution",
        extra={"dataset": plan.dataset, "written_scenes": progress.stats.written_scenes},
    )
    return ExecutionResult(
        dataset=plan.dataset,
        output_dir=plan.output_dir,
        storage_backend=plan.storage_backend,
        stats=progress.stats,
        scene_limit=progress.scene_limit,
        cleanup_summary=cleanup,
        elapsed_time_seconds=time.perf_counter() - start,
    )


def _check_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if not overwrite and next(output_dir.iterdir(), None) is not None:
        msg = (
            f"Output directory {output_dir} is not empty. "
            "Choose an empty directory or enable overwrite explicitly."
        )
        raise FileExistsError(
            msg,
        )


def run(
    request: ExecutionRequest | ExecutionPlan,
    *,
    show_progress: bool = False,
    on_progress: Callable[[Progress], None] | None = None,
) -> ExecutionResult:
    """Run a request or execute an existing plan without resolving it again."""
    resolved = request if isinstance(request, ExecutionPlan) else resolve_request(request)
    return execute_plan(resolved, show_progress=show_progress, on_progress=on_progress)


plan = resolve_request
"""Resolve a request without writing output."""
