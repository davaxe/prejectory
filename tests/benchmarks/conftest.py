from __future__ import annotations

import importlib.util
import pickle  # ruff: ignore[suspicious-pickle-import]
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest

from prejectory.io.records import SceneRecord
from tests.support import assert_scene_record_equal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from prejectory.io.base import DatasetReader


SCENES = 128
BATCH_SIZE = 8
SEED = 12345


def _make_record(scene_number: int, rng: np.random.Generator) -> SceneRecord:
    agents = int(rng.integers(8, 97))
    frames = 91
    nodes = 0 if scene_number % 8 == 0 else int(rng.integers(128, 2049))
    velocity = rng.normal(0, 5, (agents, frames, 2)).astype(np.float32)
    position = np.cumsum(velocity * np.float32(0.1), axis=1)
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1]) / np.float32(0.1)
    yaw = np.arctan2(velocity[..., 1:2], velocity[..., :1])
    features = np.concatenate((position, velocity, acceleration, yaw), axis=2)
    # Agents enter/leave the scene at different times; missing observations are zeroed.
    starts = rng.integers(0, 11, agents)
    ends = rng.integers(70, frames + 1, agents)
    time = np.arange(frames)[None, :]
    mask = (time >= starts[:, None]) & (time < ends[:, None])
    features[~mask] = 0
    source = np.arange(max(0, nodes - 1), dtype=np.int32)
    edges = np.stack((
        np.concatenate((source, source + 1)),
        np.concatenate((source + 1, source)),
    )).astype(np.int32)
    return SceneRecord(
        scene_number=scene_number,
        dataset_id=scene_number % 16,
        agent_ids=np.arange(agents, dtype=np.int64),
        agent_types=rng.integers(1, 4, agents, dtype=np.int32),
        screened_agent_mask=rng.random(agents) > 0.2,
        features=features,
        valid_mask=mask,
        map_node_positions=rng.normal(0, 50, (nodes, 2)).astype(np.float32),
        map_node_types=rng.integers(1, 5, nodes, dtype=np.int32),
        map_edge_indices=edges,
        map_edge_types=rng.integers(1, 4, 2 * max(0, nodes - 1), dtype=np.int32),
        position_offset=rng.normal(0, 1000, 2),
        prediction_origin=(11, 21, 50)[scene_number % 3],
        prediction_end=frames,
        ego_agent_id=0,
    )


def _write_export(path: Path, backend: str, records: tuple[SceneRecord, ...]) -> None:
    split = path / "unsplit"
    if backend == "pickle":
        split.mkdir(parents=True)
        for record in records:
            with (split / f"{record.scene_number:08d}.pkl").open("wb") as handle:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif backend == "mds":
        pytest.importorskip("streaming")
        from streaming import MDSWriter  # ruff: ignore[import-outside-top-level]

        from prejectory.io.encoding.mds import encode_mds_row, mds_columns  # ruff: ignore[import-outside-top-level]

        with MDSWriter(
            out=str(split), columns=mds_columns("float32"), size_limit=4 * 1024**2
        ) as writer:
            for record in records:
                writer.write(dict(encode_mds_row(record)))
    else:
        pytest.importorskip("zarr")
        from prejectory.config.models import ZarrOutputConfig  # ruff: ignore[import-outside-top-level]
        from prejectory.io.backends.zarr import (  # ruff: ignore[import-outside-top-level]
            ZarrDatasetWriter,
            _ZarrShard,  # pyright: ignore[reportPrivateUsage]
        )

        # Use the production record writer, not a simplified Zarr schema.
        for start in range(0, len(records), 32):
            shard = _ZarrShard.create(
                split / f"part-{start:05d}.zarr", record=records[start], config=ZarrOutputConfig()
            )
            for record in records[start : start + 32]:
                shard.append(record)
            shard.flush()
        ZarrDatasetWriter.finish_dataset(path, splits=None)


@dataclass
class SyntheticCorpus:
    root: Path
    records: tuple[SceneRecord, ...]
    exports: dict[tuple[str, int], tuple[Path, ...]] = field(default_factory=dict)

    def paths(self, backend: str, streams: int) -> tuple[Path, ...]:
        key = (backend, streams)
        if key not in self.exports:
            per_stream = SCENES // streams
            paths = tuple(self.root / backend / str(streams) / str(i) for i in range(streams))
            for i, path in enumerate(paths):
                _write_export(path, backend, self.records[i * per_stream : (i + 1) * per_stream])
            self.exports[key] = paths
        return self.exports[key]

    def reader(self, backend: str, streams: int = 1, *, shuffle: bool = False) -> DatasetReader:
        paths = self.paths(backend, streams)
        if backend == "mds":
            from streaming import Stream  # ruff: ignore[import-outside-top-level]

            from prejectory.io.readers.mds import MDSReader  # ruff: ignore[import-outside-top-level]

            return MDSReader(
                streams=[Stream(local=str(path), split="unsplit") for path in paths],
                shuffle=shuffle,
                shuffle_seed=SEED,
                batch_size=BATCH_SIZE,
            )
        if backend == "zarr":
            from prejectory.io.readers.zarr import ZarrReader  # ruff: ignore[import-outside-top-level]

            return ZarrReader(paths[0])
        from prejectory.io.readers.pickle import PickleReader  # ruff: ignore[import-outside-top-level]

        return PickleReader(paths[0])

    def validate(self, reader: DatasetReader) -> None:
        # Validate all arrays and metadata independently of the timed consumption.
        assert len(reader) == SCENES
        for i, expected in enumerate(self.records):
            assert_scene_record_equal(reader[i], expected)


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCorpus:
    rng = np.random.default_rng(SEED)
    return SyntheticCorpus(
        root=tmp_path_factory.mktemp("read-benchmarks"),
        records=tuple(_make_record(i, rng) for i in range(SCENES)),
    )


@pytest.fixture
def single_torch_thread() -> Iterator[None]:
    if importlib.util.find_spec("torch") is None:
        yield
        return
    import torch  # ruff: ignore[import-outside-top-level]

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
