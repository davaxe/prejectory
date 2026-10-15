"""Manifest-aware access to complete local dataset exports."""

from __future__ import annotations

import json
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from prejectory.core.categories import DatasetSplit
from prejectory.core.errors import ConfigurationError, UnsupportedStorageBackendError
from prejectory.io.base import DatasetReader, IterableDatasetReader
from prejectory.io.manifest import DatasetManifest, read_manifest
from prejectory.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


class OpenedDataset(DatasetReader[SceneRecord], IterableDatasetReader[SceneRecord]):
    """Indexed and iterable records with their manifest and selected partitions.

    Construct with :func:`open_dataset`. Iteration delegates to each backend,
    preserving MDS worker partitioning; indexing uses global record positions.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        splits: tuple[str, ...],
        readers: tuple[DatasetReader[SceneRecord], ...],
    ) -> None:
        self.manifest: DatasetManifest = manifest
        self.splits: tuple[str, ...] = splits
        self._readers: tuple[DatasetReader[SceneRecord], ...] = readers
        self._ends: tuple[int, ...] = tuple(accumulate(len(reader) for reader in readers))

    @override
    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @override
    def __iter__(self) -> Iterator[SceneRecord]:
        for reader in self._readers:
            yield from reader

    @override
    def __getitem__(self, at: int) -> SceneRecord:
        if at < 0:
            at += len(self)
        if not 0 <= at < len(self):
            raise IndexError(at)
        partition = bisect_right(self._ends, at)
        start = 0 if partition == 0 else self._ends[partition - 1]
        return self._readers[partition][at - start]


def open_dataset(
    path: str | Path,
    *,
    split: DatasetSplit | str | None = None,
) -> OpenedDataset:
    """Open a local export, detecting its backend from the manifest.

    Omit `split` to read all manifest partitions in order; use `"unsplit"`
    for unassigned records. Missing paths, splits, and count mismatches raise.
    Only open trusted pickle exports, since unpickling can execute code.

    Use backend-specific readers for remote streams or tuning options.
    """
    root = Path(path)
    manifest = read_manifest(root)
    if manifest.storage_backend not in {"pickle", "mds", "zarr"}:
        raise UnsupportedStorageBackendError(manifest.storage_backend, ("pickle", "mds", "zarr"))
    if (manifest.payload_format, manifest.payload_version) != (
        "prejectory.scene",
        1,
    ):
        msg = (
            f"Payload {manifest.payload_format!r} v{manifest.payload_version} "
            "is unsupported; expected 'prejectory.scene' v1."
        )
        raise ConfigurationError(
            msg,
        )
    splits = _select_splits(root, manifest, split)

    readers: list[DatasetReader[SceneRecord]] = []
    for name in splits:
        reader: DatasetReader[SceneRecord]
        if manifest.storage_backend == "pickle":
            from prejectory.io.readers.pickle import PickleReader  # ruff: ignore[import-outside-top-level]

            reader = PickleReader(root, split=name)
        elif manifest.storage_backend == "mds":
            # Empty MDS partitions may have no index/shards.
            if manifest.split_counts[name] == 0:
                _validate_empty_mds_split(root / name)
                continue
            from prejectory.io.readers.mds import MDSReader  # ruff: ignore[import-outside-top-level]

            reader = MDSReader(path=root, split=name)
        else:
            from prejectory.io.readers.zarr import ZarrReader  # ruff: ignore[import-outside-top-level]

            reader = ZarrReader(root, split=name)
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
