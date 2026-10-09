"""Benchmark balanced mixtures of existing MDS exports, including Torch and PyG.

Each adapter/stream-count case runs in a fresh process. Complete, shuffled
epochs use Stream(choose=...) to sample every dataset without repacking it.
Indexed reads use the same seeded indices across versions. OS caches remain
warm; imports and one warmup pass are excluded from throughput measurements.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import resource
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_io import _summary, _versions, _wrap


class _StoredSamples:
    """Expose all stored indices, independently of the sampled epoch length."""

    def __init__(self, reader: Any, size: int) -> None:  # ruff: ignore[any-type]
        self.reader = reader
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:  # ruff: ignore[any-type]
        return self.reader[index]


def _consume(
    records: Any,  # ruff: ignore[any-type]
    *,
    forecast: bool = False,
    batched: bool = False,
) -> dict[str, Any]:
    identities = Counter()
    start = time.perf_counter_ns()
    for record in records:
        if batched:
            if isinstance(record, list):
                identities.update(
                    (-1 if r.dataset_id is None else int(r.dataset_id), int(r.scene_number))
                    for r in record
                )
            else:
                identities.update(
                    zip(record.dataset_id.tolist(), record.scene_number.tolist(), strict=True)
                )
            continue
        output = record.forecast() if forecast else record
        identities[
            -1 if output.dataset_id is None else int(output.dataset_id), int(output.scene_number)
        ] += 1
    elapsed = (time.perf_counter_ns() - start) / 1e6
    count = identities.total()
    if not count:
        msg = "Benchmark received no records"
        raise RuntimeError(msg)
    datasets = Counter()
    for (dataset, _scene), n in identities.items():
        datasets[dataset] += n
    return {
        "ms_per_scene": elapsed / count,
        "count": count,
        "checksum": sum(
            (dataset * 1_000_003 + scene) * n for (dataset, scene), n in identities.items()
        ),
        "datasets": dict(datasets),
    }


def _worker(config: dict[str, Any]) -> dict[str, Any]:
    from streaming import Stream  # ruff: ignore[import-outside-top-level]

    from prejectory.io.readers.mds import MDSReader  # ruff: ignore[import-outside-top-level]

    if config["adapter"] != "native":
        import torch  # ruff: ignore[import-outside-top-level]

        torch.set_num_threads(1)
        importlib.import_module(f"prejectory.io.adapters.{config['adapter']}")
    start = time.perf_counter_ns()
    reader = MDSReader(
        streams=[
            Stream(local=path, split=config["split"], choose=config["per_stream"])
            for path in config["paths"]
        ],
        shuffle=config["shuffle"],
        shuffle_seed=config["seed"],
        batch_size=config["batch_size"],
    )
    open_ms = (time.perf_counter_ns() - start) / 1e6
    iterable = _wrap(reader, config["adapter"], config["view"], iterable=True)
    forecast = config["adapter"] == "native" and config["view"] == "forecast"
    rng = np.random.default_rng(config["seed"])
    indices = []
    offset = 0
    for path in config["paths"]:
        shards = json.loads((Path(path) / config["split"] / "index.json").read_text())["shards"]
        size = sum(shard["samples"] for shard in shards)
        indices.extend((offset + rng.choice(size, config["per_stream"], replace=False)).tolist())
        offset += size
    rng.shuffle(indices)
    indexed = _wrap(_StoredSamples(reader, offset), config["adapter"], config["view"])
    cases = {
        "random": lambda: (indexed[i] for i in indices),
        "iterate": lambda: iter(iterable),
    }
    if config["workers"] is not None and config["adapter"] != "native":
        module = (
            importlib.import_module("prejectory.io.adapters.pyg")
            if config["adapter"] == "pyg"
            else None
        )
        from benchmark_io import _identity_batch  # ruff: ignore[import-outside-top-level]

        collate = (
            getattr(
                module,
                "collate_hetero_with_time_padding"
                if config["view"] == "full"
                else "collate_forecast_hetero_with_time_padding",
            )
            if module
            else _identity_batch
        )
        loader = torch.utils.data.DataLoader(
            iterable,
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            collate_fn=collate,
            persistent_workers=bool(config["workers"]),
            multiprocessing_context="spawn" if config["workers"] else None,
        )

        cases["loader"] = lambda: iter(loader)
        if config["workers"]:
            # Spawn before parent-side iteration creates Mosaic thread locks.
            cases = {"loader": cases.pop("loader"), **cases}
    results = {}
    for name, records in cases.items():
        warmup = _consume(records(), forecast=forecast, batched=name == "loader")
        passes = [
            _consume(records(), forecast=forecast, batched=name == "loader")
            for _ in range(config["repeats"])
        ]
        for result in [warmup, *passes]:
            if result["count"] != len(indices):
                msg = f"{name}: received {result['count']} records, expected {len(indices)}"
                raise RuntimeError(msg)
            if result["datasets"] != warmup["datasets"]:
                msg = f"{name}: dataset sampling counts changed: {result['datasets']}"
                raise RuntimeError(msg)
        results[name] = {
            "ms_per_scene": _summary([p["ms_per_scene"] for p in passes]),
            "passes": passes,
        }
    return {
        **config,
        "reader_source": sys.modules[MDSReader.__module__].__file__,
        "open_ms": open_ms,
        "stored_scenes": offset,
        "epoch_scenes": len(reader),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }


def main() -> None:  # ruff: ignore[complex-structure]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--datasets", help="Comma-separated dataset directory names; default all")
    parser.add_argument("--streams", default="1,4,all")
    parser.add_argument("--adapter", default="native,torch,pyg")
    parser.add_argument("--view", choices=("full", "forecast"), default="full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-stream", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers", type=int, help="Also time an iterable DataLoader (persistent if >0)"
    )
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(json.load(sys.stdin))))
        return
    if args.root is None:
        parser.error("root is required")
    if min(args.per_stream, args.repeats, args.batch_size) < 1 or (
        args.workers is not None and args.workers < 0
    ):
        parser.error(
            "per-stream, repeats and batch-size must be positive; workers cannot be negative"
        )
    paths = sorted(p for p in args.root.iterdir() if (p / args.split / "index.json").is_file())
    if args.datasets:
        paths = [args.root / name for name in args.datasets.split(",")]
    if not paths:
        parser.error("No dataset splits found")
    adapters = args.adapter.split(",")
    if not set(adapters) <= {"native", "torch", "pyg"}:
        parser.error("adapter must be native,torch,pyg")
    counts = [len(paths) if count == "all" else int(count) for count in args.streams.split(",")]
    if any(count < 1 or count > len(paths) for count in counts):
        parser.error("stream counts must be between 1 and the number of datasets")
    results = []
    for count, adapter in product(counts, adapters):
        config = {
            "paths": [str(p.resolve()) for p in paths[:count]],
            "adapter": adapter,
            "view": args.view,
            "split": args.split,
            "per_stream": args.per_stream,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "shuffle": args.shuffle,
            "seed": args.seed,
        }
        process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, __file__, "--worker"],
            input=json.dumps(config),
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            msg = f"Benchmark failed for {count} streams/{adapter}:\n{process.stderr}"
            raise RuntimeError(msg)
        result = json.loads(process.stdout)
        results.append(result)
        medians = " ".join(
            f"{name}={value['ms_per_scene']['median']:.4f}"
            for name, value in result["results"].items()
        )
        print(f"{count} streams {adapter}: {medians} ms/scene", flush=True)
    output = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _versions(),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
