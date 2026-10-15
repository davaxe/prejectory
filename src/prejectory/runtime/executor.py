# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
"""Internal runtime executors."""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prejectory.runtime.accounting import (
    CleanupAccumulator,
    LocalRunAccounting,
    SharedRunAccounting,
    iter_scenes_from_source,
)
from prejectory.runtime.processor import RuntimeProcessor
from prejectory.runtime.state import Progress, SharedResources

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from multiprocessing.context import BaseContext
    from multiprocessing.pool import Pool
    from multiprocessing.synchronize import Barrier, Event

    from prejectory.core.scene import Scene
    from prejectory.io.base import DatasetWriter, WriterProvider
    from prejectory.processing.loading.models import DatasetSource
    from prejectory.runtime.types import CleanupSummary, ExecutionPlan

_ctx: WorkerRuntime

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerRuntime:
    """Runtime objects available inside one worker process."""

    shared: SharedResources
    worker_id: int
    processor: RuntimeProcessor | None = None
    writer: DatasetWriter | None = None
    finish_barrier: Barrier | None = None


@contextmanager
def open_executor(plan: ExecutionPlan) -> Generator[SequentialExecutor | ParallelExecutor]:
    """Open the executor and dataset resources for one plan."""
    with plan._descriptor.open_resources(plan.input_dir, plan._loader) as map_provider:
        logger.debug("Opening executor", extra={"dataset": plan.dataset})
        loader = plan._descriptor.build_loader(
            root=plan.input_dir,
            request=plan._loader,
            map_provider=map_provider,
        )
        processor = RuntimeProcessor.from_plan(plan, loader)
        yield _build_executor(plan, processor)


def _build_executor(
    plan: ExecutionPlan,
    processor: RuntimeProcessor,
) -> SequentialExecutor | ParallelExecutor:
    if plan.parallel:
        logger.debug("Using parallel executor", extra={"dataset": plan.dataset})
        return ParallelExecutor(
            processor,
            workers=plan.config.runtime.jobs,
            chunksize=plan.config.runtime.chunksize,
            limit=plan.limit,
        )
    logger.debug("Using sequential executor", extra={"dataset": plan.dataset})
    return SequentialExecutor(processor, limit=plan.limit)


class SequentialExecutor:
    """Single-process executor for internal runtime execution.

    Parameters
    ----------
    processor: RuntimeProcessor
        The runtime processor to execute, containing logic and configurations.
    limit: int | None, optional
        An optional limit on the total number of scenes to select. If None, no
        limit will be applied. Default is None.
    """

    def __init__(self, processor: RuntimeProcessor, *, limit: int | None = None) -> None:
        self._processor: RuntimeProcessor = processor
        self._screening_enabled: bool = processor.screening_enabled()
        self._total_sources: int | None = processor.total_sources()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False
        self._accounting: LocalRunAccounting = LocalRunAccounting(
            limit=limit,
            update_event=self._update_event,
        )

    def execute(self, writer_provider: WriterProvider) -> Progress:
        """Process all selected sources with one writer."""
        writer = writer_provider.open_worker(0)
        try:
            for scene in self._iter_scenes():
                writer.write(scene)
                self._accounting.record_written(scene.split_assignment)
        finally:
            try:
                writer.finish_local()
            finally:
                writer_provider.finish_final()

        return self.snapshot()

    def snapshot(self) -> Progress:
        """Return current sequential progress."""
        return self._accounting.snapshot(
            running=self._running,
            total_sources=self._total_sources,
            active_workers=1 if self._running else 0,
            screening_enabled=self._screening_enabled,
        )

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return final cleanup statistics."""
        return self._accounting.cleanup_summary()

    def changed(self) -> threading.Event:
        """Return the event signaled after progress changes."""
        return self._update_event

    def _iter_scenes(self) -> Iterator[Scene]:
        self._running = True
        self._update_event.set()

        try:
            for source in self._processor.iter_sources():
                if self._accounting.limit_reached():
                    break
                yield from iter_scenes_from_source(self._processor, source, self._accounting)
        finally:
            self._running = False
            self._update_event.set()


class ParallelExecutor:
    """Parallel executor for internal runtime execution.

    Parameters
    ----------
    processor: RuntimeProcessor
        The runtime processor to execute, containing logic and cofigurations.
    chunksize: int | None, optional
        The number of sources to process in each worker batch. If None, an
        optimal chunksize will be estimated based on a simple heuristic. Default
        is None.
    workers: int | None, optional
        The number of worker processes to use for parallel execution. If None,
        the number of CPU cores will be used. Default is None.
    limit: int | None, optional
        An optional limit on the total number of scenes to select across all
        workers. If None, no limit will be applied. Default is None.
    """

    def __init__(
        self,
        processor: RuntimeProcessor,
        *,
        chunksize: int | None = None,
        workers: int | None = None,
        limit: int | None = None,
        mp_context: BaseContext | None = None,
    ) -> None:
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for parallel execution."
            raise ValueError(msg)
        self._processor: RuntimeProcessor = processor
        total_sources = processor.total_sources()
        self._chunksize: int = chunksize or self._optimal_chunksize(total_sources, workers)
        self._limit: int | None = limit
        self._processes: int | None = workers
        self._num_sources: int | None = total_sources
        self._screening_enabled: bool = processor.screening_enabled()
        self._running: bool = False
        self._cleanup_accumulator: CleanupAccumulator = CleanupAccumulator()
        self._mp_context: BaseContext = mp_context or mp.get_context("spawn")
        self._shared: SharedResources = SharedResources.create(
            scene_limit=limit,
            mp_context=self._mp_context,
        )

    def execute(self, writer_provider: WriterProvider) -> Progress:
        """Process selected sources across the worker pool."""
        for cleanup_summary in self._execute_parallel(writer_provider):
            self._cleanup_accumulator.merge(cleanup_summary)
        writer_provider.finish_final()
        return self.snapshot()

    def snapshot(self) -> Progress:
        """Return current shared progress."""
        return self._shared.progress.snapshot(
            running=self._running,
            total_sources=self._num_sources,
            scene_limit=self._limit,
            screening_enabled=self._screening_enabled,
        )

    def changed(self) -> Event:
        """Return the shared event signaled after progress changes."""
        return self._shared.progress.update_event

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return merged cleanup statistics from all workers."""
        return self._cleanup_accumulator.freeze()

    @staticmethod
    def _process_fn_write(source: DatasetSource[Any]) -> CleanupSummary | None:
        if _ctx.writer is None:
            msg = "DatasetWriter was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.processor is None:
            msg = "Runtime processor was not initialized for this worker process."
            raise ValueError(msg)

        accounting = SharedRunAccounting(
            progress=_ctx.shared.progress,
            limit=_ctx.shared.scene_limit,
        )
        if accounting.limit_reached():
            return None

        try:
            for scene in iter_scenes_from_source(_ctx.processor, source, accounting):
                _ctx.writer.write(scene)
                accounting.record_written(scene.split_assignment)
        finally:
            _ctx.writer.flush_local()

        return accounting.cleanup_summary()

    def _execute_parallel(
        self,
        writer_provider: WriterProvider,
    ) -> Iterator[CleanupSummary | None]:
        self._shared.reset()
        worker_count = self._processes or mp.cpu_count()
        finish_barrier = self._mp_context.Barrier(worker_count)
        self._running = True
        self.changed().set()
        pool: Pool | None = None
        completed = False
        try:
            pool = self._mp_context.Pool(
                worker_count,
                initializer=_init_write_worker,
                initargs=(self._shared, self._processor, writer_provider, finish_barrier),
            )
            yield from pool.imap_unordered(
                self._process_fn_write,
                self._processor.iter_sources(),
                self._chunksize,
            )
            # Each finish task waits at the barrier, so no worker can take a
            # second task before every worker has closed its own writer.
            _ = pool.map(_finish_write_worker, range(worker_count), chunksize=1)
            completed = True
        finally:
            if pool is not None:
                if completed:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
            self._running = False
            self.changed().set()

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        if num_sources is None:
            return 1
        process_count = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, process_count * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)


def _init_write_worker(
    shared: SharedResources,
    processor: RuntimeProcessor,
    writer_provider: WriterProvider,
    finish_barrier: Barrier,
) -> None:
    global _ctx  # ruff: ignore[global-statement]
    worker_id = shared.next_worker()
    shared.progress.worker_started()
    _ctx = WorkerRuntime(
        shared=shared,
        worker_id=worker_id,
        processor=processor,
        finish_barrier=finish_barrier,
    )
    try:
        _ctx.writer = writer_provider.open_worker(worker_id)
    except Exception:
        _ctx.shared.progress.worker_stopped()
        raise


def _finish_write_worker(_task_id: int) -> None:
    try:
        if _ctx.writer is not None:
            _ctx.writer.finish_local()
    finally:
        _ctx.writer = None
        _ctx.shared.progress.worker_stopped()
        if _ctx.finish_barrier is not None:
            _ = _ctx.finish_barrier.wait(timeout=300)
