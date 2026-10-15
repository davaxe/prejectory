"""Framework-neutral readers for pickled raw scene-record exports."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from prejectory.io.base import DatasetReader, split_directory_name
from prejectory.io.records import SceneRecord

if TYPE_CHECKING:
    from prejectory.core.categories import DatasetSplit


class PickleReader(DatasetReader[SceneRecord]):
    """Read `SceneRecord` objects written by the pickle backend."""

    def __init__(
        self,
        path: str | Path,
        split: DatasetSplit | str | None = None,
    ) -> None:
        self._path: Path = Path(path) / split_directory_name(split)
        if not self._path.exists():
            msg = f"Dataset split directory does not exist: {self._path}"
            raise FileNotFoundError(msg)
        if not self._path.is_dir():
            msg = f"Dataset split path is not a directory: {self._path}"
            raise NotADirectoryError(msg)
        self._files: tuple[Path, ...] = tuple(sorted(self._path.glob("*.pkl")))

    @override
    def __len__(self) -> int:
        return len(self._files)

    @override
    def __getitem__(self, at: int) -> SceneRecord:
        with self._files[at].open("rb") as file:
            record = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]
        if not isinstance(record, SceneRecord):
            msg = f"Expected pickled SceneRecord, but got {type(record).__name__}."
            raise TypeError(msg)
        return record
