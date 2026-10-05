from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, runtime_checkable

from typing_extensions import TypeVar, override

from prejectory.core.categories import DatasetSplit
from prejectory.core.scene import Scene
from prejectory.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


RecordT = TypeVar("RecordT", default=SceneRecord)


RecordTransform: TypeAlias = Callable[[SceneRecord], RecordT]
"""Callable that converts a canonical `SceneRecord` into a persisted payload.

This is the preferred customization hook for user-defined persisted payload
layouts because it preserves Prejectory's standard output semantics before
materializing the custom payload.
"""


SceneTransform: TypeAlias = Callable[[Scene], RecordT]
"""Callable that converts a runtime `Scene` directly into a persisted payload.

This is an advanced escape hatch for users who intentionally want to bypass
`SceneRecord` encoding. Callers using this hook own schema conversion,
dtype policy, recentering, map resolution, and reader compatibility.
"""


class StorageBackend(StrEnum):
    """Supported persisted storage backends."""

    MDS = "mds"
    """MDS storage backend.

    ??? warning "Extra dependencies"
        Using MDS requires installing the `prejectory[mds]` extra:

        ``sh
        pip install prejectory[mds]
        ``
    """
    PICKLE = "pickle"
    """Pickle storage backend. Requires no extra dependencies."""
    ZARR = "zarr"
    """Chunked Zarr storage backend with random scene access."""
    NULL = "null"
    """Null storage backend that discards all data. Useful for testing."""


class DatasetReader(ABC, Generic[RecordT]):
    """Abstract base class for scene readers."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of scene records."""

    def __iter__(self) -> Iterator[RecordT]:
        """Iterate over decoded scene records."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def __getitem__(self, at: int) -> RecordT:
        """Return a single decoded scene record."""


class IterableDatasetReader(ABC, Generic[RecordT]):
    """Abstract base class for iterable scene readers.

    This is a specialization of `DatasetReader` for backends that support
    streaming data without random access.

    """

    @abstractmethod
    def __iter__(self) -> Iterator[RecordT]:
        """Iterate over decoded scene records."""

    def __len__(self) -> int:
        """Return the number of scene records, if known."""
        msg = f"{self.__class__.__name__} does not implement __len__"
        raise NotImplementedError(msg)


@runtime_checkable
class DatasetWriter(Protocol):
    """Protocol for writing processed scenes to persisted storage."""

    def write(self, scene: Scene) -> None:
        """Write one processed scene or raise if it cannot be committed."""
        ...

    def finish_local(self) -> None:
        """Finalize worker-local state once the current worker is done."""
        _ = self

    def flush_local(self) -> None:
        """Flush worker-local state to persisted storage without finalizing."""
        _ = self


class WriterProvider(ABC):
    """Provider that owns writer lifecycle for one execution run."""

    @abstractmethod
    def open_worker(self, worker_id: int) -> DatasetWriter:
        """Create the writer used by one worker."""

    def finish_final(self) -> None:
        """Finalize dataset-wide writer state after all workers finish."""
        _ = self


def _noop_finish_final() -> None:
    return


@dataclass(frozen=True, slots=True)
class WorkerWriterProvider(WriterProvider):
    """Pickleable writer provider backed by top-level callables."""

    create_worker: Callable[[int], DatasetWriter]
    finalize: Callable[[], None] = _noop_finish_final

    @override
    def open_worker(self, worker_id: int) -> DatasetWriter:
        """Create the writer used by one worker."""
        return self.create_worker(worker_id)

    @override
    def finish_final(self) -> None:
        """Finalize dataset-wide writer state after all workers finish."""
        self.finalize()


def split_directory_name(split: DatasetSplit | str | None) -> str:
    """Return the storage subdirectory name for one dataset split."""
    if split is None:
        return "unsplit"
    return split.value if isinstance(split, DatasetSplit) else str(split)


def validate_transform_choice(
    *,
    record_transform: RecordTransform[object] | None,
    scene_transform: SceneTransform[object] | None,
) -> None:
    """Validate that at most one output customization hook is configured."""
    if record_transform is not None and scene_transform is not None:
        msg = "Use either `record_transform` or `scene_transform`, not both."
        raise ValueError(msg)
