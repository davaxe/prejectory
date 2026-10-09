# ruff: file-ignore[import-outside-top-level, private-member-access]
# pyright: reportPrivateUsage=false
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _write_streams(root: Path, compression: str | None = None) -> list[Path]:
    streaming = pytest.importorskip("streaming")
    paths: list[Path] = []
    for stream, count in enumerate((7, 4, 5)):
        path = root / str(stream)
        with streaming.MDSWriter(
            out=str(path / "unsplit"), columns={"integer": "int"}, compression=compression
        ) as writer:
            for index in range(count):
                writer.write({"integer": 100 * stream + index})
        paths.append(path)
    return paths


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_local_epochs_match_mosaic(
    tmp_path: Path, *, shuffle: bool, batch_size: int, weighted: bool
) -> None:
    streaming = pytest.importorskip("streaming")
    from prejectory.io.readers.mds import MDSReader

    paths = _write_streams(tmp_path / "local")
    _ = shutil.copytree(tmp_path / "local", tmp_path / "upstream")
    chooses = (3, 9, 2) if weighted else (None, None, None)
    kwargs: dict[str, Any] = {"shuffle": shuffle, "batch_size": batch_size, "shuffle_seed": 37}
    local = MDSReader(
        streams=[
            streaming.Stream(local=str(p), split="unsplit", choose=n)
            for p, n in zip(paths, chooses, strict=True)
        ],
        convert_raw=dict,
        **kwargs,
    )
    upstream = streaming.StreamingDataset(
        streams=[
            streaming.Stream(local=str(tmp_path / "upstream" / p.name), split="unsplit", choose=n)
            for p, n in zip(paths, chooses, strict=True)
        ],
        **kwargs,
    )
    assert local._backend._local_iteration
    assert len(local) == len(upstream)
    for _ in range(3):
        assert list(local) == list(upstream)
    assert not hasattr(local._backend, "_executor")


@pytest.mark.parametrize("mode", ["disabled", "compressed", "remote", "cache_limit"])
def test_local_iteration_falls_back_when_preparation_needed(tmp_path: Path, mode: str) -> None:
    streaming = pytest.importorskip("streaming")
    from prejectory.io.readers.mds import MDSReader

    paths = _write_streams(tmp_path / "local", "zstd:3" if mode == "compressed" else None)
    kwargs: dict[str, Any] = {}
    if mode == "remote":
        _ = shutil.copytree(tmp_path / "local", tmp_path / "remote")
    if mode == "cache_limit":
        kwargs["cache_limit"] = "1mb"
    reader = MDSReader(
        streams=[
            streaming.Stream(
                local=str(p),
                split="unsplit",
                remote=str(tmp_path / "remote" / p.name) if mode == "remote" else None,
            )
            for p in paths
        ],
        convert_raw=dict,
        local_iteration=mode != "disabled",
        **kwargs,
    )
    assert not reader._backend._local_iteration
    assert [row["integer"] for row in reader] == [*range(7), *range(100, 104), *range(200, 205)]
    assert hasattr(reader._backend, "_executor")


def test_local_iteration_invalidates_interrupted_epoch(tmp_path: Path) -> None:
    streaming = pytest.importorskip("streaming")
    from prejectory.io.readers.mds import MDSReader

    reader = MDSReader(
        streams=[streaming.Stream(local=str(p), split="unsplit") for p in _write_streams(tmp_path)],
        convert_raw=dict,
    )
    old = iter(reader)
    assert next(old) == {"integer": 0}
    assert len(list(reader)) == 16
    assert list(old) == []


def test_local_iteration_resumes_mosaic_checkpoint(tmp_path: Path) -> None:
    streaming = pytest.importorskip("streaming")
    from prejectory.io.readers.mds import MDSReader

    paths = _write_streams(tmp_path / "source")
    _ = shutil.copytree(tmp_path / "source", tmp_path / "restored")
    upstream = streaming.StreamingDataset(
        streams=[streaming.Stream(local=str(p), split="unsplit") for p in paths],
        shuffle=True,
        shuffle_seed=37,
        batch_size=1,
    )
    epoch = iter(upstream)
    for _ in range(5):
        next(epoch)
    state = upstream.state_dict(5, from_beginning=True)
    expected = list(epoch)
    reader = MDSReader(
        streams=[
            streaming.Stream(local=str(tmp_path / "restored" / p.name), split="unsplit")
            for p in paths
        ],
        convert_raw=dict,
        shuffle=True,
        batch_size=1,
    )
    reader._backend.load_state_dict(state)
    assert list(reader) == expected
