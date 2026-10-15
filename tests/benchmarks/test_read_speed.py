from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from prejectory.io.base import IterableDatasetReader
from tests.benchmarks.conftest import BATCH_SIZE, SCENES, SEED, SyntheticCorpus

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from pytest_benchmark.fixture import BenchmarkFixture

    from prejectory.io.base import DatasetReader

pytestmark = [
    pytest.mark.slow,
    pytest.mark.performance,
    pytest.mark.usefixtures("single_torch_thread"),
]


@dataclass(frozen=True)
class ReadResult:
    scene_numbers: tuple[int, ...]
    agents: int


def _consume(
    records: Iterable[Any], adapter: str, view: str, *, batched: bool = False
) -> ReadResult:
    scenes: list[int] = []
    agents = 0
    for record in records:
        if batched and adapter == "torch":
            for item in record:
                scenes.append(int(item.scene_number))
                agents += int(item.agent_ids.shape[0])
        elif adapter == "pyg":
            if batched:
                scenes.extend(record.scene_number.tolist())
            else:
                scenes.append(int(record.scene_number))
            agents += int(record["agent"].num_nodes)
        else:
            output = record.forecast() if adapter == "native" and view == "forecast" else record
            scenes.append(int(output.scene_number))
            agents += int(output.agent_ids.shape[0])
    return ReadResult(tuple(scenes), agents)


def _wrap(reader: DatasetReader, adapter: str, view: str, *, iterable: bool = True) -> Any:  # ruff: ignore[any-type]
    if adapter == "native":
        return reader
    pytest.importorskip("torch")
    if adapter == "pyg":
        pytest.importorskip("torch_geometric")
    module = importlib.import_module(f"prejectory.io.adapters.{adapter}")
    name = ("Iterable" if iterable else "") + ("Torch" if adapter == "torch" else "Hetero")
    name += "SceneDataset" if view == "full" else "ForecastDataset"
    return getattr(module, name)(reader)


def _identity_collate(records: list[Any]) -> list[Any]:
    # Scene records have variable agent/map dimensions; preserve their full payloads.
    return records


@contextmanager
def _loader(reader: DatasetReader, adapter: str, view: str, workers: int) -> Generator[Any]:
    # Only MDS partitions iterable readers itself. Map adapters let DataLoader
    # partition Pickle/Zarr indices rather than duplicating the epoch per worker.
    dataset = _wrap(reader, adapter, view, iterable=isinstance(reader, IterableDatasetReader))
    import torch  # ruff: ignore[import-outside-top-level]

    collate = _identity_collate
    if adapter == "pyg":
        module = importlib.import_module("prejectory.io.adapters.pyg")
        collate = getattr(
            module,
            "collate_hetero_with_time_padding"
            if view == "full"
            else "collate_forecast_hetero_with_time_padding",
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate,
        persistent_workers=bool(workers),
        multiprocessing_context="spawn" if workers else None,
        prefetch_factor=2 if workers else None,
    )
    try:
        yield loader
    finally:
        # Stop persistent workers even when assertions fail; no workers survive the test.
        iterator = cast("Any", loader)._iterator  # ruff: ignore[private-member-access]
        if iterator is not None:
            iterator._shutdown_workers()  # ruff: ignore[private-member-access]


def _measure(
    benchmark: BenchmarkFixture,
    records: Iterable[Any],
    corpus: SyntheticCorpus,
    adapter: str,
    view: str,
    *,
    batched: bool = False,
) -> None:
    benchmark.extra_info.update({
        "scenes_per_round": SCENES,
        "agents_per_round": sum(r.agent_ids.size for r in corpus.records),
        "cache": "warm OS cache; backend caches retained across rounds",
        "batch_size": BATCH_SIZE if batched else None,
    })
    # Writing, reader creation, worker startup and cache warmup are outside timing.
    warmup = _consume(records, adapter, view, batched=batched)
    result = cast("ReadResult", benchmark(_consume, records, adapter, view, batched=batched))
    for output in (warmup, result):
        assert sorted(output.scene_numbers) == list(range(SCENES))
        assert output.agents == benchmark.extra_info["agents_per_round"]


@pytest.mark.parametrize("backend", ["pickle", "mds", "zarr"])
@pytest.mark.parametrize("pattern", ["sequential", "random"])
@pytest.mark.benchmark(group="indexed-read", min_rounds=5, max_time=0.5)
def test_indexed_read_speed(
    benchmark: BenchmarkFixture, synthetic_corpus: SyntheticCorpus, backend: str, pattern: str
) -> None:
    reader = synthetic_corpus.reader(backend)
    synthetic_corpus.validate(reader)
    indices = list(range(SCENES))
    if pattern == "random":
        np.random.default_rng(SEED).shuffle(indices)

    def read() -> ReadResult:
        return _consume((reader[i] for i in indices), "native", "full")

    benchmark.extra_info["scenes_per_round"] = SCENES
    result = cast("ReadResult", benchmark(read))
    assert result.scene_numbers == tuple(indices)
    assert result.agents == sum(r.agent_ids.size for r in synthetic_corpus.records)


@pytest.mark.parametrize("backend", ["pickle", "mds", "zarr"])
@pytest.mark.parametrize("pattern", ["sequential", "random"])
@pytest.mark.benchmark(group="indexed-read-fresh-reader")
def test_fresh_reader_read_speed(
    benchmark: BenchmarkFixture, synthetic_corpus: SyntheticCorpus, backend: str, pattern: str
) -> None:
    reader = synthetic_corpus.reader(backend)
    synthetic_corpus.validate(reader)
    indices = list(range(SCENES))
    if pattern == "random":
        np.random.default_rng(SEED).shuffle(indices)

    def setup() -> tuple[tuple[DatasetReader], dict[str, Any]]:
        return (synthetic_corpus.reader(backend),), {}

    def read(fresh_reader: DatasetReader) -> ReadResult:
        return _consume((fresh_reader[i] for i in indices), "native", "full")

    benchmark.extra_info.update({
        "scenes_per_round": SCENES,
        "cache": "fresh backend; warm OS cache",
    })
    # Reopen before each round without timing construction. This includes shard
    # offset/chunk loads that retained readers (especially Zarr) can otherwise skip.
    result = cast(
        "ReadResult",
        benchmark.pedantic(read, setup=setup, rounds=5, iterations=1, warmup_rounds=1),
    )
    assert result.scene_numbers == tuple(indices)
    assert result.agents == sum(r.agent_ids.size for r in synthetic_corpus.records)


@pytest.mark.parametrize("backend", ["pickle", "mds", "zarr"])
@pytest.mark.parametrize("view", ["full", "forecast"])
@pytest.mark.benchmark(group="native-iteration", min_rounds=5, max_time=0.5)
def test_iteration_read_speed(
    benchmark: BenchmarkFixture, synthetic_corpus: SyntheticCorpus, backend: str, view: str
) -> None:
    reader = synthetic_corpus.reader(backend)
    synthetic_corpus.validate(reader)
    _measure(benchmark, reader, synthetic_corpus, "native", view)


@pytest.mark.parametrize("backend", ["pickle", "mds", "zarr"])
@pytest.mark.parametrize("adapter", ["torch", "pyg"])
@pytest.mark.parametrize("view", ["full", "forecast"])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.benchmark(group="dataloader", min_rounds=5, max_time=0.5)
def test_dataloader_read_speed(
    benchmark: BenchmarkFixture,
    synthetic_corpus: SyntheticCorpus,
    backend: str,
    adapter: str,
    view: str,
    workers: int,
) -> None:
    reader = synthetic_corpus.reader(backend)
    synthetic_corpus.validate(reader)
    with _loader(reader, adapter, view, workers) as loader:
        _measure(benchmark, loader, synthetic_corpus, adapter, view, batched=True)


@pytest.mark.parametrize("streams", [1, 4, 16])
@pytest.mark.parametrize("adapter", ["native", "torch", "pyg"])
@pytest.mark.parametrize("view", ["full", "forecast"])
@pytest.mark.benchmark(group="mds-streams", min_rounds=5, max_time=0.5)
def test_mds_stream_read_speed(
    benchmark: BenchmarkFixture,
    synthetic_corpus: SyntheticCorpus,
    streams: int,
    adapter: str,
    view: str,
) -> None:
    reader = synthetic_corpus.reader("mds", streams, shuffle=True)
    synthetic_corpus.validate(reader)
    _measure(benchmark, _wrap(reader, adapter, view), synthetic_corpus, adapter, view)


@pytest.mark.parametrize("adapter", ["torch", "pyg"])
@pytest.mark.parametrize("view", ["full", "forecast"])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.benchmark(group="mds-stream-dataloader", min_rounds=5, max_time=0.5)
def test_mds_stream_dataloader_read_speed(
    benchmark: BenchmarkFixture,
    synthetic_corpus: SyntheticCorpus,
    adapter: str,
    view: str,
    workers: int,
) -> None:
    reader = synthetic_corpus.reader("mds", 16, shuffle=True)
    synthetic_corpus.validate(reader)
    with _loader(reader, adapter, view, workers) as loader:
        _measure(benchmark, loader, synthetic_corpus, adapter, view, batched=True)
