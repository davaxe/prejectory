# ruff: file-ignore[undocumented-public-method]
"""Runtime progress models and multiprocessing execution state."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from prejectory.core.categories import DatasetSplit

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event


class SplitCounts(TypedDict, total=True):
    unsplit: int
    train: int
    val: int
    test: int


# Keep a direct Literal: ty rejects PEP 695 aliases as writable TypedDict keys.
SplitKey = Literal["unsplit", "train", "val", "test"]


def empty_split_counts() -> SplitCounts:
    """Return zero-initialized split counts."""
    return {"unsplit": 0, "train": 0, "val": 0, "test": 0}


def split_key(split: DatasetSplit | None) -> SplitKey:
    """Return the split-count key for a split assignment."""
    if split is None:
        return "unsplit"
    return split.value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ScreeningProgress:
    """Screening counters for candidate scenes."""

    enabled: bool = False
    passed: int = 0
    rejected: int = 0


@dataclass(frozen=True, slots=True)
class CleanupProgress:
    """Lightweight cleanup counters for live progress reporting."""

    rows_total: int = 0
    rows_removed: int = 0
    agents_total: int = 0
    agents_removed: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    """Counters shared by live progress snapshots and final execution results."""

    processed_sources: int = 0
    candidate_scenes: int = 0
    written_scenes: int = 0
    split_counts: SplitCounts = field(default_factory=empty_split_counts)
    cleanup: CleanupProgress = field(default_factory=CleanupProgress)
    screening: ScreeningProgress = field(default_factory=ScreeningProgress)


@dataclass(frozen=True, slots=True)
class Progress:
    """Immutable live progress snapshot.

    `stats` is the canonical counter payload. Convenience properties expose the
    same values directly for concise display and API code.
    """

    running: bool = False
    total_sources: int | None = None
    scene_limit: int | None = None
    active_workers: int = 0
    stats: ExecutionStats = field(default_factory=ExecutionStats)

    @property
    def split_counts(self) -> SplitCounts:
        return self.stats.split_counts

    @property
    def cleanup(self) -> CleanupProgress:
        return self.stats.cleanup

    @property
    def screening(self) -> ScreeningProgress:
        return self.stats.screening


@dataclass(slots=True)
class ProgressState:
    """Multiprocessing-safe mutable progress state.

    This class owns shared progress mutation. Executors should expose immutable
    `Progress` snapshots instead of reading counters directly.
    """

    active_workers: Synchronized[int]
    source_counter: Synchronized[int]
    candidate_scene_counter: Synchronized[int]
    screening_passed_counter: Synchronized[int]
    screening_rejected_counter: Synchronized[int]
    selected_scene_counter: Synchronized[int]
    written_scene_counter: Synchronized[int]
    cleanup_rows_total_counter: Synchronized[int]
    cleanup_rows_removed_counter: Synchronized[int]
    cleanup_agents_total_counter: Synchronized[int]
    cleanup_agents_removed_counter: Synchronized[int]
    unsplit_counter: Synchronized[int]
    train_counter: Synchronized[int]
    val_counter: Synchronized[int]
    test_counter: Synchronized[int]
    update_event: Event

    @classmethod
    def create(cls, mp_context: BaseContext | None = None) -> ProgressState:
        ctx = mp_context or mp.get_context()
        return cls(
            active_workers=ctx.Value("i", 0),
            source_counter=ctx.Value("i", 0),
            candidate_scene_counter=ctx.Value("i", 0),
            screening_passed_counter=ctx.Value("i", 0),
            screening_rejected_counter=ctx.Value("i", 0),
            selected_scene_counter=ctx.Value("i", 0),
            written_scene_counter=ctx.Value("i", 0),
            cleanup_rows_total_counter=ctx.Value("i", 0),
            cleanup_rows_removed_counter=ctx.Value("i", 0),
            cleanup_agents_total_counter=ctx.Value("i", 0),
            cleanup_agents_removed_counter=ctx.Value("i", 0),
            unsplit_counter=ctx.Value("i", 0),
            train_counter=ctx.Value("i", 0),
            val_counter=ctx.Value("i", 0),
            test_counter=ctx.Value("i", 0),
            update_event=ctx.Event(),
        )

    def reset(self) -> None:
        """Reset all progress counters and clear the update event."""
        for counter in self._counters():
            _set_counter(counter, 0)
        self.update_event.clear()

    def snapshot(
        self,
        *,
        running: bool,
        total_sources: int | None,
        scene_limit: int | None,
        screening_enabled: bool,
    ) -> Progress:
        """Return an immutable point-in-time progress snapshot."""
        return Progress(
            running=running,
            total_sources=total_sources,
            scene_limit=scene_limit,
            active_workers=self.active_workers.value,
            stats=self.execution_stats(screening_enabled=screening_enabled),
        )

    def execution_stats(self, *, screening_enabled: bool) -> ExecutionStats:
        """Return an immutable snapshot of shared execution counters."""
        return ExecutionStats(
            processed_sources=self.source_counter.value,
            candidate_scenes=self.candidate_scene_counter.value,
            written_scenes=self.written_scene_counter.value,
            split_counts=self.split_counts(),
            cleanup=CleanupProgress(
                rows_total=self.cleanup_rows_total_counter.value,
                rows_removed=self.cleanup_rows_removed_counter.value,
                agents_total=self.cleanup_agents_total_counter.value,
                agents_removed=self.cleanup_agents_removed_counter.value,
            ),
            screening=ScreeningProgress(
                enabled=screening_enabled,
                passed=self.screening_passed_counter.value,
                rejected=self.screening_rejected_counter.value,
            ),
        )

    def record_processed_source(self) -> None:
        """Record that one source has finished processing."""
        self._increment_and_notify(self.source_counter)

    def record_candidate_scene(self) -> None:
        """Record one generated candidate scene."""
        self._increment_and_notify(self.candidate_scene_counter)

    def record_screening_result(self, *, passed: bool) -> None:
        """Record one screening decision."""
        if passed:
            self._increment_and_notify(self.screening_passed_counter)
        else:
            self._increment_and_notify(self.screening_rejected_counter)

    def claim_selected_scene(self, limit: int | None = None) -> int | None:
        """Claim the next selected scene number, respecting the optional limit."""
        with self.selected_scene_counter.get_lock():
            if limit is not None and self.selected_scene_counter.value >= limit:
                return None
            scene_number = self.selected_scene_counter.value
            self.selected_scene_counter.value += 1
        self.update_event.set()
        return scene_number

    def selected_scene_limit_reached(self, limit: int | None = None) -> bool:
        """Return whether the selected-scene counter has reached `limit`."""
        if limit is None:
            return False
        with self.selected_scene_counter.get_lock():
            return self.selected_scene_counter.value >= limit

    def record_cleanup(
        self,
        *,
        rows_total: int,
        rows_removed: int,
        agents_total: int,
        agents_removed: int,
    ) -> None:
        """Record cleanup row/agent totals for one candidate scene."""
        for counter, amount in (
            (self.cleanup_rows_total_counter, rows_total),
            (self.cleanup_rows_removed_counter, rows_removed),
            (self.cleanup_agents_total_counter, agents_total),
            (self.cleanup_agents_removed_counter, agents_removed),
        ):
            _add_counter(counter, amount)
        self.update_event.set()

    def record_written_scene(self, split: DatasetSplit | None) -> None:
        """Record one successfully committed scene and its output split."""
        _add_counter(self.written_scene_counter, 1)
        _add_counter(self._split_counter(split), 1)
        self.update_event.set()

    def split_counts(self) -> SplitCounts:
        """Return current output split counts."""
        return {
            "unsplit": self.unsplit_counter.value,
            "train": self.train_counter.value,
            "val": self.val_counter.value,
            "test": self.test_counter.value,
        }

    def worker_started(self) -> None:
        """Record one active worker process."""
        self._increment_and_notify(self.active_workers)

    def worker_stopped(self) -> None:
        """Record one worker process stopping."""
        _add_counter(self.active_workers, -1)
        self.update_event.set()

    def _counters(self) -> tuple[Synchronized[int], ...]:
        return (
            self.active_workers,
            self.source_counter,
            self.candidate_scene_counter,
            self.screening_passed_counter,
            self.screening_rejected_counter,
            self.selected_scene_counter,
            self.written_scene_counter,
            self.cleanup_rows_total_counter,
            self.cleanup_rows_removed_counter,
            self.cleanup_agents_total_counter,
            self.cleanup_agents_removed_counter,
            self.unsplit_counter,
            self.train_counter,
            self.val_counter,
            self.test_counter,
        )

    def _split_counter(self, split: DatasetSplit | None) -> Synchronized[int]:
        match split:
            case None:
                return self.unsplit_counter
            case DatasetSplit.TRAIN:
                return self.train_counter
            case DatasetSplit.VAL:
                return self.val_counter
            case DatasetSplit.TEST:
                return self.test_counter

    def _increment_and_notify(self, counter: Synchronized[int]) -> None:
        _add_counter(counter, 1)
        self.update_event.set()


@dataclass(slots=True)
class SharedResources:
    """Shared state passed to runtime worker processes."""

    progress: ProgressState
    next_worker_id: Synchronized[int]
    scene_limit: int | None = None

    @classmethod
    def create(
        cls,
        *,
        scene_limit: int | None = None,
        mp_context: BaseContext | None = None,
    ) -> SharedResources:
        ctx = mp_context or mp.get_context()
        return cls(
            progress=ProgressState.create(ctx),
            next_worker_id=ctx.Value("i", 0),
            scene_limit=scene_limit,
        )

    def reset(self) -> None:
        _set_counter(self.next_worker_id, 0)
        self.progress.reset()

    def next_worker(self) -> int:
        """Allocate the next worker identifier."""
        _add_counter(self.next_worker_id, 1)
        return self.next_worker_id.value


def _set_counter(counter: Synchronized[int], value: int) -> None:
    with counter.get_lock():
        counter.value = value


def _add_counter(counter: Synchronized[int], amount: int) -> None:
    with counter.get_lock():
        counter.value += amount
