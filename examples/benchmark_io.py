"""Compare processed exports through native, Torch, and PyG read paths.

python examples/benchmark_io.py data/waymo-mds data/zarr --backend mds,zarr \
    --adapter native,torch,pyg --view full,forecast --output .cache/io.json

Each dataset/adapter/view runs in a fresh subprocess, isolating Mosaic shared
memory and Python caches. Imports are excluded from opening time. First reads
are cold-ish only: this tool never evicts the OS page cache. Forecast views
slice a fully loaded record; they do not claim storage-level partial I/O.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import islice, product
from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _wrap(reader: Any, adapter: str, view: str, *, iterable: bool = False) -> Any:  # ruff: ignore[any-type]
    if adapter == "native":
        return reader
    module = importlib.import_module(f"prejectory.io.adapters.{adapter}")
    name = ("Torch" if adapter == "torch" else "Hetero") + (
        "SceneDataset" if view == "full" else "ForecastDataset"
    )
    return getattr(module, ("Iterable" if iterable else "") + name)(reader)


def _run_pass(
    dataset: Any,  # ruff: ignore[any-type]
    indices: list[int],
    *,
    forecast: bool,
    iterate: bool,
) -> tuple[float, int]:
    checksum = 0
    start = time.perf_counter_ns()
    records = islice(iter(dataset), len(indices)) if iterate else (dataset[i] for i in indices)
    count = 0
    for record in records:
        output = record.forecast() if forecast else record
        checksum += int(output.scene_number)
        count += 1
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if count != len(indices):
        msg = f"Expected {len(indices)} records, received {count}"
        raise RuntimeError(msg)
    return elapsed / count, checksum


def _identity_batch(records: list[Any]) -> list[Any]:
    # Torch scene records have variable agent/map sizes; preserve the list.
    return records


def _loader_passes(dataset: Any, config: dict[str, Any], count: int) -> dict[str, Any]:  # ruff: ignore[any-type]
    import torch  # ruff: ignore[import-outside-top-level]

    collate = _identity_batch
    if config["adapter"] == "pyg":
        module = importlib.import_module("prejectory.io.adapters.pyg")
        collate = getattr(
            module,
            "collate_hetero_with_time_padding"
            if config["view"] == "full"
            else "collate_forecast_hetero_with_time_padding",
        )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(range(count))),
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        collate_fn=collate,
        # Spawn exercises reader serialization and avoids forking live Mosaic
        # preparation threads. Worker startup is included in each pass.
        multiprocessing_context="spawn" if config["workers"] else None,
    )
    samples = []
    checksums = []
    for _ in range(config["repeats"]):
        start = time.perf_counter_ns()
        checksum = 0
        seen = 0
        for batch in loader:
            if config["adapter"] == "pyg":
                checksum += int(batch.scene_number.sum())
                seen += batch.num_graphs
            else:
                checksum += sum(int(record.scene_number) for record in batch)
                seen += len(batch)
        if seen != count:
            msg = f"DataLoader returned {seen} scenes, expected {count}"
            raise RuntimeError(msg)
        samples.append((time.perf_counter_ns() - start) / 1e6 / count)
        checksums.append(checksum)
    return {"ms_per_scene": _summary(samples), "checksums": checksums}


def _worker(config: dict[str, Any]) -> dict[str, Any]:
    from prejectory.io.dataset import open_dataset  # ruff: ignore[import-outside-top-level]

    importlib.import_module(f"prejectory.io.readers.{config['backend']}")
    if config["adapter"] != "native":
        import torch  # ruff: ignore[import-outside-top-level]

        torch.set_num_threads(config["threads"])
        importlib.import_module(f"prejectory.io.adapters.{config['adapter']}")
    start = time.perf_counter_ns()
    reader = open_dataset(config["path"], split=config["split"])
    dataset = _wrap(reader, config["adapter"], config["view"])
    open_ms = (time.perf_counter_ns() - start) / 1e6
    count = min(config["count"], len(reader))
    if count < 1:
        msg = "Cannot benchmark an empty dataset"
        raise ValueError(msg)
    forecast = config["adapter"] == "native" and config["view"] == "forecast"
    first_ms, _ = _run_pass(dataset, [0], forecast=forecast, iterate=False)
    rng = np.random.default_rng(config["seed"])
    sequential = list(range(count))
    indices = {
        "sequential": sequential,
        "random": rng.choice(len(reader), count, replace=False).tolist(),
        "repeated": [0] * count,
        "iterate": sequential,
    }
    iterable = _wrap(reader, config["adapter"], config["view"], iterable=True)
    timings: dict[str, list[float]] = {name: [] for name in config["patterns"]}
    checksums: dict[str, list[int]] = {name: [] for name in config["patterns"]}
    for _ in range(config["repeats"]):
        for name in config["patterns"]:
            elapsed, checksum = _run_pass(
                iterable if name == "iterate" else dataset,
                indices[name],
                forecast=forecast,
                iterate=name == "iterate",
            )
            timings[name].append(elapsed)
            checksums[name].append(checksum)
    # Measure conversion without backend I/O using the identical sampled records.
    records = [reader[i] for i in indices["random"]]
    conversion = _wrap(records, config["adapter"], config["view"])
    conversion_ms = [
        _run_pass(conversion, sequential, forecast=forecast, iterate=False)[0]
        for _ in range(config["repeats"])
    ]
    sizes = [record.features.nbytes for record in records]
    loader_results = (
        _loader_passes(dataset, config, count)
        if config.get("batch_size") and config["adapter"] != "native"
        else None
    )
    return {
        **config,
        "total_scenes": len(reader),
        "sample_scenes": count,
        "open_ms": open_ms,
        "first_sample_ms": first_ms,
        "ms_per_scene": {name: _summary(values) for name, values in timings.items()},
        "conversion_ms_per_scene": _summary(conversion_ms),
        "checksums": checksums,
        "sample_feature_bytes": _summary(sizes),
        "dataloader": loader_results,
    }


def _versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "zarr", "mosaicml-streaming", "torch", "torch-geometric"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def main() -> None:  # ruff: ignore[complex-structure]
    """Run isolated backend/adapter comparisons and preserve raw timing samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument("--backend", default="mds,zarr,pickle")
    parser.add_argument("--adapter", default="native,torch,pyg")
    parser.add_argument("--view", default="full")
    parser.add_argument("--patterns", default="sequential,random,repeated,iterate")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0, help="Also time adapter DataLoaders")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(json.load(sys.stdin))))
        return
    if min(args.count, args.repeats, args.threads) < 1:
        parser.error("count, repeats, and threads must be positive")
    if min(args.batch_size, args.workers) < 0:
        parser.error("batch-size and workers cannot be negative")
    for field, allowed in (
        ("backend", {"mds", "zarr", "pickle"}),
        ("adapter", {"native", "torch", "pyg"}),
        ("view", {"full", "forecast"}),
        ("patterns", {"sequential", "random", "repeated", "iterate"}),
    ):
        values = getattr(args, field).split(",")
        if not set(values) <= allowed:
            parser.error(f"Invalid --{field}; choose from {sorted(allowed)}")
        setattr(args, field, values)
    settings = {
        name: value
        for name, value in vars(args).items()
        if name not in {"datasets", "output", "worker"}
    }
    results = []
    for path in args.datasets:
        manifest = json.loads((path / "manifest.json").read_text())
        backend = manifest["storage_backend"]
        if backend not in args.backend:
            continue
        for adapter, view in product(args.adapter, args.view):
            config = {
                **settings,
                "path": str(path.resolve()),
                "backend": backend,
                "adapter": adapter,
                "view": view,
            }
            process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                [sys.executable, __file__, "--worker"],
                input=json.dumps(config),
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode:
                msg = f"Benchmark failed for {config}:\n{process.stderr}"
                raise RuntimeError(msg)
            result = json.loads(process.stdout)
            results.append(result)
            medians = " ".join(
                f"{name}={stats['median']:.4f}" for name, stats in result["ms_per_scene"].items()
            )
            print(f"{path} {adapter}/{view}: {medians} ms/scene", flush=True)
    if not results:
        parser.error("No datasets matched the selected backends")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "versions": _versions(),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
