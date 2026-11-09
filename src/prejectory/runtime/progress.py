"""Rich-backed progress helpers for the optional CLI."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, override

import rich.progress as rp
from rich import box
from rich.console import Group, RenderableType, RichCast
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from prejectory.runtime.observer import ProgressSource, observe_execution
from prejectory.runtime.state import Progress, SplitCounts

if TYPE_CHECKING:
    from collections.abc import Callable


class _ExecutorDisplay(RichCast):
    """Custom renderable that stacks a progress bar and runtime statistics."""

    def __init__(self) -> None:
        self.progress: rp.Progress = rp.Progress(
            rp.SpinnerColumn(spinner_name=_random_spinner_name(), style="bold cyan"),
            rp.TextColumn("[bold white]Processing"),
            rp.BarColumn(bar_width=40, style="dim", complete_style="green"),
            rp.TaskProgressColumn(),
            rp.MofNCompleteColumn(),
            rp.TextColumn("•"),
            rp.TimeElapsedColumn(),
            rp.TextColumn("•"),
            rp.TimeRemainingColumn(),
            expand=False,
        )
        self.task_id: rp.TaskID = self.progress.add_task("run", total=0)
        self.state: Progress = Progress()

    def update(self, progress_state: Progress) -> None:
        completed, total = _progress_bar_counts(progress_state)
        self.state = progress_state
        self.progress.update(self.task_id, completed=completed, total=total)

    @override
    def __rich__(self) -> Panel:
        layout: list[RenderableType] = [self.progress, ""]
        layout.append(_centered_markup(self._main_stats_markup()))

        if self.state.screening.enabled:
            layout.append(_centered_markup(self._screening_markup()))

        if (cleanup_markup := self._cleanup_markup()) is not None:
            layout.append(_centered_markup(cleanup_markup))

        if (splits_markup := self._splits_markup()) is not None:
            layout.append(_centered_markup(splits_markup))

        return Panel(Group(*layout), title_align="left", box=box.MINIMAL, expand=False)

    def _main_stats_markup(self) -> str:
        source_part = _source_progress_text(
            processed=self.state.stats.processed_sources,
            total=self.state.total_sources,
        )
        return (
            f"[bold cyan]Workers:[/bold cyan] {self._visible_workers}"
            f"    [bold blue]Sources:[/bold blue] {source_part}"
            f"    [bold magenta]Scenes written:[/bold magenta] {self.state.stats.written_scenes}"
        )

    def _screening_markup(self) -> str:
        candidates = self.state.stats.candidate_scenes
        screening = self.state.screening
        return (
            "[bold yellow]Screening:[/bold yellow] "
            f"{screening.passed} / {candidates} passed "
            f"({_percent(screening.passed, candidates):.1f}%)"
            f" • rejected: {screening.rejected}"
        )

    def _cleanup_markup(self) -> str | None:
        cleanup = self.state.cleanup
        if cleanup.rows_removed <= 0 and cleanup.agents_removed <= 0:
            return None

        return (
            "[bold red]Cleanup:[/bold red] "
            f"{cleanup.rows_removed} / {cleanup.rows_total} rows removed"
            f" ({_percent(cleanup.rows_removed, cleanup.rows_total):.1f}%) • "
            f"{cleanup.agents_removed} / {cleanup.agents_total} agents removed"
            f" ({_percent(cleanup.agents_removed, cleanup.agents_total):.1f}%)"
        )

    def _splits_markup(self) -> str | None:
        nonzero_splits = _nonzero_splits(self.state.split_counts)
        if not nonzero_splits:
            return None

        total = sum(nonzero_splits.values())
        splits = " • ".join(
            _format_split(name, count, total) for name, count in nonzero_splits.items()
        )
        return f"[bold green]Splits:[/bold green] {splits}"

    @property
    def _visible_workers(self) -> int:
        return self.state.active_workers if self.state.running else 0


def execute_with_rich_progress[T](
    progress: ProgressSource,
    run: Callable[[], T],
    *,
    on_progress: Callable[[Progress], None] | None = None,
) -> T:
    """Render progress and deliver optional callbacks through the same observer."""
    display = _ExecutorDisplay()

    def update(snapshot: Progress) -> None:
        display.update(snapshot)
        if on_progress is not None:
            on_progress(snapshot)

    with Live(display, refresh_per_second=4, transient=False):
        return observe_execution(progress, run, update)


def _progress_bar_counts(progress: Progress) -> tuple[int, int | None]:
    """Return completed/total values for the main progress bar."""
    if progress.scene_limit is not None:
        completed = progress.stats.written_scenes
        total: int | None = progress.scene_limit
    else:
        completed = progress.stats.processed_sources
        total = progress.total_sources

    if not progress.running and (total is None or completed < total):
        total = completed

    return completed, total


def _centered_markup(markup: str) -> Text:
    text = Text.from_markup(markup)
    text.justify = "center"
    return text


def _format_split(name: str, count: int, total: int) -> str:
    return f"{name}: {count} ({_percent(count, total):.1f}%)"


def _nonzero_splits(split_counts: SplitCounts) -> dict[str, int]:
    return {
        name: count for name, count in split_counts.items() if isinstance(count, int) and count > 0
    }


def _source_progress_text(*, processed: int, total: int | None) -> str:
    if total is None:
        return str(processed)
    return f"{processed} / {total}"


def _percent(part: int, total: int) -> float:
    return (part / total) * 100 if total > 0 else 0.0


def _random_spinner_name() -> str:
    number = random.randint(1, 12)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    return f"dots{number}" if number > 1 else "dots"
