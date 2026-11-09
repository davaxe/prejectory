"""Framework-neutral observation of a synchronous execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from prejectory.runtime.state import Progress


class ProgressSource(Protocol):
    """Execution source exposing current counters."""

    def snapshot(self) -> Progress:
        """Return a point-in-time snapshot."""
        ...


def observe_execution[T](
    source: ProgressSource,
    run: Callable[[], T],
    callback: Callable[[Progress], None] | None,
) -> T:
    """Observe on a background thread, propagating callback errors after work stops."""
    if callback is None:
        return run()
    stop = threading.Event()
    errors: list[BaseException] = []

    def observe() -> None:
        try:
            callback(source.snapshot())
            while not stop.wait(0.1):
                callback(source.snapshot())
            callback(source.snapshot())
        except BaseException as exc:  # ruff: ignore[blind-except]
            errors.append(exc)

    thread = threading.Thread(target=observe, name="prejectory-progress", daemon=True)
    thread.start()
    try:
        result = run()
    finally:
        stop.set()
        thread.join()
    if errors:
        raise errors[0]
    return result
