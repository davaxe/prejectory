"""No-op writer backend.

The `null` backend executes the full runtime, counts accepted scenes, and
intentionally skips all persisted scene output. It is useful for tests, dry
runs, configuration validation, and measuring pipeline cost without storage
I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final, override

from prejectory.io.base import DatasetWriter

if TYPE_CHECKING:
    from prejectory.core.scene import Scene


@final
class NullWriter(DatasetWriter):
    """No-op writer used by tests and dry-run execution paths."""

    @override
    def write(self, scene: Scene) -> None:
        _ = scene
