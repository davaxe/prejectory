"""Benchmark complete SceneRecord reads from a Prejectory Zarr export.

Example: .venv/bin/python benchmark_zarr.py data/zarr --count 100 --repeats 5
The first pass is only cold-ish: this tool does not evict the OS page cache.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from prejectory.io.readers import ZarrReader


def read_pass(reader: ZarrReader, indices: list[int]) -> tuple[float, int, int]:
    """Time full record construction and touch all returned array payloads."""
    payload = 0
    checksum = 0
    start = time.perf_counter()
    for index in indices:
        record = reader[index]
        arrays = (
            record.position_offset,
            record.agent_ids,
            record.agent_types,
            record.screened_agent_mask,
            record.features,
            record.valid_mask,
            record.map_node_positions,
            record.map_node_types,
            record.map_edge_indices,
            record.map_edge_types,
        )
        payload += sum(array.nbytes for array in arrays)
        checksum += int(record.scene_number)
    return time.perf_counter() - start, payload, checksum


def summarize(values: list[float]) -> dict[str, float]:
    """Return timing statistics in milliseconds per scene."""
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main() -> None:
    """Run repeated sequential and randomized complete-record reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--json", type=Path, help="Write machine-readable measurements")
    args = parser.parse_args()
    if args.count < 1 or args.repeats < 1:
        parser.error("--count and --repeats must be positive")

    probe = ZarrReader(args.dataset, split=args.split)
    total = len(probe)
    if not 0 <= args.start < total:
        parser.error(f"--start must be in [0, {total - 1}]")
    count = min(args.count, total - args.start)
    sequential = list(range(args.start, args.start + count))
    random_indices = (
        np.random.default_rng(args.seed).choice(total, size=count, replace=False).tolist()
    )
    del probe

    measurements: dict[str, list[float]] = {
        "open_ms": [],
        "sequential_first_ms": [],
        "random_first_ms": [],
        "sequential_repeat_ms": [],
        "random_repeat_ms": [],
    }
    signatures: dict[str, tuple[int, int]] = {}
    for _ in range(args.repeats):
        start = time.perf_counter()
        reader = ZarrReader(args.dataset, split=args.split)
        measurements["open_ms"].append(1000 * (time.perf_counter() - start))
        for name, indices in (
            ("sequential_first_ms", sequential),
            ("random_first_ms", random_indices),
            ("sequential_repeat_ms", sequential),
            ("random_repeat_ms", random_indices),
        ):
            elapsed, payload, checksum = read_pass(reader, indices)
            signature = payload, checksum
            if name in signatures and signatures[name] != signature:
                msg = f"Inconsistent read result in {name}"
                raise RuntimeError(msg)
            signatures[name] = signature
            measurements[name].append(1000 * elapsed / count)

    result = {
        "dataset": str(args.dataset),
        "split": args.split,
        "total_scenes": total,
        "sample_scenes": count,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": {name: summarize(values) for name, values in measurements.items()},
        "samples": measurements,
        "signatures": signatures,
    }
    for name, summary in result["results"].items():
        print(  # ruff: ignore[print]
            f"{name:22s} median {summary['median']:8.3f}  "
            f"min {summary['min']:8.3f}  max {summary['max']:8.3f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
