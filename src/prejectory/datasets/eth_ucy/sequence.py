"""Select and reconstruct ETH/UCY recordings from four-column text files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from prejectory.core.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

DOMAIN_RECORDINGS = {
    "eth": ("biwi_eth",),
    "hotel": ("biwi_hotel",),
    "univ": ("students001", "students003"),
    "zara1": ("crowds_zara01",),
    "zara2": ("crowds_zara02",),
}
EXCLUDED_RECORDINGS = {"uni_examples", "crowds_zara03"}
_RECORDINGS = {name for members in DOMAIN_RECORDINGS.values() for name in members}
_NATIVE_DIRS = {"train", "val", "test"}


def discover_recordings(root: Path, domain: str) -> dict[str, tuple[Path, ...]]:
    """Group copies and split fragments by recording, preferring whole inputs."""
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.txt")):
        name = path.stem
        for suffix in ("_train", "_val", "_test"):
            if name.endswith(suffix):
                name = name.removesuffix(suffix)
                break
        if name in EXCLUDED_RECORDINGS:
            continue
        if name not in _RECORDINGS:
            msg = f"Unknown ETH/UCY recording: {path}. Expected mapped four-column text files."
            raise ConfigurationError(msg)
        found.setdefault(name, []).append(path)

    recordings: dict[str, tuple[Path, ...]] = {}
    for name in DOMAIN_RECORDINGS[domain]:
        files = found.get(name, [])
        if not files:
            msg = f"Missing required recording {name!r} for domain {domain!r}."
            raise ConfigurationError(msg)
        originals = [
            path
            for path in files
            if path.stem == name and _NATIVE_DIRS.isdisjoint(path.relative_to(root).parts[:-1])
        ]
        recordings[name] = tuple(originals or files)
    return recordings


def read_recording(
    files: tuple[Path, ...], *, recording_index: int = 0, recording_count: int = 1
) -> pl.DataFrame:
    """Deduplicate before normalizing time and namespacing pedestrian IDs."""
    parts: list[pl.DataFrame] = []
    for path in files:
        try:
            values = np.loadtxt(path, dtype=np.float64, ndmin=2)
        except (ValueError, OSError) as exc:
            msg = f"Cannot read ETH/UCY recording {path}: {exc}"
            raise ConfigurationError(msg) from exc
        if not values.size or values.shape[1] != 4 or not np.isfinite(values).all():
            msg = f"Expected nonempty finite four-column frame/id/x/y data: {path}."
            raise ConfigurationError(msg)
        if (values[:, :2] < 0).any() or (values[:, :2] != np.floor(values[:, :2])).any():
            msg = f"Frame and pedestrian IDs must be nonnegative integers: {path}."
            raise ConfigurationError(msg)
        parts.append(pl.DataFrame(values, schema=["frame", "id", "x", "y"], orient="row"))

    frame = pl.concat(parts).with_columns(pl.col("frame", "id").cast(pl.Int64))
    frame = frame.unique().sort("frame", "id")
    if frame.select("frame", "id").is_duplicated().any():
        msg = f"Conflicting duplicate coordinates in recording files: {files}."
        raise ConfigurationError(msg)
    relative_time = pl.col("frame") - pl.col("frame").min()
    if frame.filter(relative_time % 10 != 0).height:
        msg = f"Expected ten original frame units per annotation timestep: {files}."
        raise ConfigurationError(msg)
    return frame.with_columns(
        (relative_time // 10).cast(pl.Int32),
        (pl.col("id") * recording_count + recording_index).cast(pl.Int32),
    )
