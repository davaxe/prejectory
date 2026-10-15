# Read benchmarks

Benchmarks Pickle, MDS, and Zarr reads, Torch/PyG DataLoaders, and combined MDS
streams using 128 synthetic scenes. No external datasets are needed.

```sh
uv sync --group test --extra mds --extra zarr --extra torch --extra pyg
uv run --no-sync pytest tests/benchmarks --benchmark-only \
  --benchmark-save baseline --benchmark-histogram=.benchmarks/baseline
```

Compare after a change:

```sh
uv run --no-sync pytest tests/benchmarks --benchmark-only \
  --benchmark-save candidate --benchmark-compare \
  --benchmark-histogram=.benchmarks/comparison
```

Generate charts and a CSV from saved runs without rerunning the tests:

```sh
uv run --no-sync pytest-benchmark compare \
  --histogram=.benchmarks/saved --csv=.benchmarks/results
```

Open the generated `.benchmarks/*.svg` charts in a browser for timing distributions
and tooltips, or the CSV in a spreadsheet. Charts are grouped by workload.

Timings cover a 128-scene pass with warm OS caches; setup and worker startup
are excluded. Divide by 128 for per-scene latency. Results are saved in
`.benchmarks/`. Use `-k` to select cases or `--benchmark-disable` to check
correctness without timing.
