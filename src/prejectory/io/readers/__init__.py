"""Framework-neutral readers for persisted dataset outputs.

All readers in this package share the same output contract: they yield
[`SceneRecord`][prejectory.io.SceneRecord] instances, which are
storage-agnostic in-memory representations of scene records.

## Import guide

``python
from prejectory.io.readers import DatasetReader, IterableDatasetReader
from prejectory.io.readers import MDSReader, MDSReaderInitArgs, PickleReader
``

## Related modules

- [`prejectory.io`][] for storage contracts and export configuration
- [`prejectory.io.adapters`][] for optional adapter layers built on top of
  readers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prejectory.core.optional import lazy_dir, resolve_lazy_export

if TYPE_CHECKING:
    from prejectory.io.base import DatasetReader, IterableDatasetReader
    from prejectory.io.readers.mds import MDSReader, MDSReaderInitArgs
    from prejectory.io.readers.pickle import PickleReader
    from prejectory.io.readers.zarr import ZarrReader

__all__ = [
    "DatasetReader",
    "IterableDatasetReader",
    "MDSReader",
    "MDSReaderInitArgs",
    "PickleReader",
    "ZarrReader",
]

__lazy_exports__: dict[str, tuple[str, str]] = {
    "IterableDatasetReader": ("prejectory.io.base", "IterableDatasetReader"),
    "DatasetReader": ("prejectory.io.base", "DatasetReader"),
    "MDSReader": ("prejectory.io.readers.mds", "MDSReader"),
    "MDSReaderInitArgs": ("prejectory.io.readers.mds", "MDSReaderInitArgs"),
    "PickleReader": ("prejectory.io.readers.pickle", "PickleReader"),
    "ZarrReader": ("prejectory.io.readers.zarr", "ZarrReader"),
}


def __getattr__(name: str) -> object:
    """Resolve optional reader exports lazily."""
    return resolve_lazy_export(globals(), __lazy_exports__, module_name=__name__, name=name)


def __dir__() -> list[str]:
    """Expose lazy reader exports during interactive discovery."""
    return lazy_dir(globals(), exported_names=list(__lazy_exports__))
