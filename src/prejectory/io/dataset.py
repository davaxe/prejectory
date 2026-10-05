"""Manifest-aware access to complete local dataset exports."""

from __future__ import annotations

import json
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar, override

from prejectory.core.categories import DatasetSplit
from prejectory.core.errors import ConfigurationError, UnsupportedStorageBackendError
from prejectory.io.base import DatasetReader, IterableDatasetReader
from prejectory.io.manifest import DatasetManifest, read_manifest
from prejectory.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

RecordT = TypeVar("RecordT", default=SceneRecord)


class OpenedDataset(DatasetReader[RecordT], IterableDatasetReader[RecordT], Generic[RecordT]):
    """Indexed and iterable records with their manifest and selected partitions.

    Construct with :func:`open_dataset`. Iteration delegates to each backend,
    preserving MDS worker partitioning; indexing uses global record positions.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        splits: tuple[str, ...],
        readers: tuple[DatasetReader[RecordT], ...],
    ) -> None:
        self.manifest: DatasetManifest = manifest
        self.splits: tuple[str, ...] = splits
        self._readers: tuple[DatasetReader[RecordT], ...] = readers
        self._ends: tuple[int, ...] = tuple(accumulate(len(reader) for reader in readers))

    @override
    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @override
    def __iter__(self) -> Iterator[RecordT]:
        for reader in self._readers:
            yield from reader

    @override
    def __getitem__(self, at: int) -> RecordT:
        if at < 0:
            at += len(self)
        if not 0 <= at < len(self):
            raise IndexError(at)
        partition = bisect_right(self._ends, at)
        start = 0 if partition == 0 else self._ends[partition - 1]
        return self._readers[partition][at - start]


class _DecodedReader(DatasetReader[RecordT]):
    def __init__(self, reader: DatasetReader[Any], decoder: Callable[[Any], RecordT]) -> None:
        self._reader: DatasetReader[Any] = reader
        self._decoder: Callable[[Any], RecordT] = decoder

    @override
    def __len__(self) -> int:
        return len(self._reader)

    @override
    def __iter__(self) -> Iterator[RecordT]:
        for raw in self._reader:
            yield self._decoder(raw)

    @override
    def __getitem__(self, at: int) -> RecordT:
        return self._decoder(self._reader[at])


@overload
def open_dataset(
    path: str | Path,
    *,
    split: DatasetSplit | str | None = None,
    decoder: None = None,
) -> OpenedDataset[SceneRecord]: ...


@overload
def open_dataset(
    path: str | Path,
    *,
    split: DatasetSplit | str | None = None,
    decoder: Callable[[Any], RecordT],
) -> OpenedDataset[RecordT]: ...


def open_dataset(
    path: str | Path,
    *,
    split: DatasetSplit | str | None = None,
    decoder: Callable[[Any], Any] | None = None,
) -> OpenedDataset[Any]:
    """Open a local export, detecting its backend from the manifest.

    Omit `split` to read all manifest partitions in order; use `"unsplit"`
    for unassigned records. Missing paths, splits, and count mismatches raise.
    Custom formats require `decoder`: it receives the unpickled payload or
    raw MDS row. Check `read_manifest(path).payload_format` and
    `payload_version` when choosing that decoder. Only open trusted pickle
    exports, since unpickling can execute code.

    Use backend-specific readers for remote streams or tuning options.
    """
    root = Path(path)
    manifest = read_manifest(root)
    if manifest.storage_backend not in {"pickle", "mds", "zarr"}:
        raise UnsupportedStorageBackendError(manifest.storage_backend, ("pickle", "mds", "zarr"))
    if decoder is None and (manifest.payload_format, manifest.payload_version) != (
        "prejectory.scene",
        1,
    ):
        msg = (
            f"Payload {manifest.payload_format!r} v{manifest.payload_version} "
            "requires an explicit decoder."
        )
        raise ConfigurationError(
            msg,
        )
    splits = _select_splits(root, manifest, split)

    readers: list[DatasetReader[Any]] = []
    for name in splits:
        reader: DatasetReader[Any]
        if manifest.storage_backend == "pickle":
            from prejectory.io.readers.pickle import PickleReader  # ruff: ignore[import-outside-top-level]

            reader = PickleReader(
                root, split=name, record_type=SceneRecord if decoder is None else object
            )
            if decoder is not None:
                reader = _DecodedReader(reader, decoder)
        elif manifest.storage_backend == "mds":
            # Empty MDS partitions may have no index/shards.
            if manifest.split_counts[name] == 0:
                _validate_empty_mds_split(root / name)
                continue
            from prejectory.io.readers.mds import MDSReader  # ruff: ignore[import-outside-top-level]

            reader = (
                MDSReader(path=root, split=name)
                if decoder is None
                else MDSReader(
                    path=root,
                    split=name,
                    convert_raw=decoder,
                )
            )
        else:
            from prejectory.io.readers.zarr import ZarrReader  # ruff: ignore[import-outside-top-level]

            reader = ZarrReader(root, split=name)
            if decoder is not None:
                reader = _DecodedReader(reader, decoder)
        if len(reader) != manifest.split_counts[name]:
            msg = (
                f"Split {name!r} has {len(reader)} records; "
                f"manifest declares {manifest.split_counts[name]}."
            )
            raise ConfigurationError(
                msg,
            )
        readers.append(reader)
    return OpenedDataset(manifest, splits, tuple(readers))


def _select_splits(
    root: Path,
    manifest: DatasetManifest,
    split: DatasetSplit | str | None,
) -> tuple[str, ...]:
    """Validate the selected partitions before constructing any backend readers."""
    if not manifest.splits:
        msg = "Manifest has no output partitions; the export may be incomplete."
        raise ConfigurationError(msg)
    selected = split.value if isinstance(split, DatasetSplit) else split
    splits = manifest.splits if selected is None else (selected,)
    for name in splits:
        if name not in manifest.split_counts:
            msg = (
                f"Unknown output split {name!r}. "
                f"Available splits: {', '.join(manifest.splits) or 'none'}."
            )
            raise ConfigurationError(
                msg,
            )
        directory = root / name
        if not directory.exists():
            msg = f"Dataset split directory does not exist: {directory}"
            raise FileNotFoundError(msg)
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    return splits


def _validate_empty_mds_split(directory: Path) -> None:
    """MDS cannot open an empty index; verify it without constructing a reader."""
    index = directory / "index.json"
    if not index.exists():
        return
    payload = json.loads(index.read_text(encoding="utf-8"))
    if any(shard["samples"] for shard in payload["shards"]):
        msg = f"Split {directory.name!r} contains MDS samples; manifest declares zero records."
        raise ConfigurationError(msg)
