# MDS read optimization

Branch: `perf/mds-read-optimization`, base: `91a1ee5`.

Retained progression commits:

| Commit | Change |
|---|---|
| `090b74d` | Cross-backend/native/Torch/PyG benchmarks and repacking tool |
| `21f87d6` | Larger default shard-preparation window |
| `9f3ce59` | Reusable per-shard column codecs |
| `1db2b30` | DataLoader batching/spawn benchmarks |
| `916235e` | Waymo/ETH validation and comparison results |
| `469e6fa` | Single-pass variable-size header parsing |

## Data and method

The configured `$TRAJ_DATA` contained a legacy NuScenes manifest but no usable
processed shard index. The available `data/zarr` Waymo export contains 1,000
scenes (91 frames, seven features, varying agent/map counts). Repacked it with
`examples/repack_dataset.py` into `data/waymo-mds` (uncompressed, 64 MiB shards)
and `data/waymo-pickle`. All 1,000 records matched the source, including dtypes
and shapes. Source data was preserved. These are local benchmark derivatives.

`benchmark_io.py` accepts multiple export roots and filters `--backend`.
Native, Torch and PyG map-style and iterable adapters are exercised; `--view
forecast` includes the forecast conversion. Forecast reads materialize the
full stored scene, not partial storage I/O. Each case runs in a fresh process;
imports are excluded from opening latency. Timing samples, min/median/max,
opening/first-read latency, conversion-only cost and feature sizes are saved
as JSON. Torch uses one intra-op thread. OS caches are **not** flushed.
Host CPU: Intel Xeon E-2144G @ 3.60 GHz. Package/Python/platform versions are
recorded in each JSON result.

Standard command (five passes, 300 scenes, fixed random seed):

```sh
.venv/bin/python benchmark_io.py data/waymo-mds data/waymo-pickle data/zarr \
  --count 300 --repeats 5 --output results.json
```

## Progressions

### Benchmark infrastructure — `090b74d`

Baseline median milliseconds per scene (sequential / random / repeated / iteration):

| Backend | Native | Torch | PyG |
|---|---|---|---|
| MDS | .1840 / .1817 / .1531 / 1.0065 | .2557 / .2537 / .2148 / 1.0361 | .3427 / .3417 / .3032 / 1.0977 |
| Pickle | .0734 / .0740 / .0451 / .0725 | .1258 / .1262 / .0903 / .1234 | .2158 / .2194 / .1765 / .2119 |
| Zarr | .6192 / 4.7518 / .0404 / .4776 | .6809 / 4.8588 / .0845 / .5265 | .7972 / 4.9894 / .1714 / .6497 |

See `results/mds-baseline*.json` for distributions and forecast measurements.

### Keep shard preparation ahead of iteration — `21f87d6`

Hypothesis: default `batch_size=1` implicitly reduces Mosaic's predownload
window to eight samples. Profiling found preparation/ready-thread waits and
file-lock activity dominating iteration. Default to `max(64, 8 * batch_size)`;
explicit predownload values remain authoritative.

Two independent runs gave iteration medians (native / Torch / PyG):

- Before: 1.0065 / 1.0361 / 1.0977 ms.
- After: .4314 / .5629 / .6790 ms.
- Repeat: .4263 / .5492 / .6721 ms.
- Reduction (repeat): 57.6% / 47.0% / 38.8%.

Random/repeated reads were approximately unchanged. Native sequential passes
in the mixed-pattern benchmark increased to .23-.25 ms; interrupted iteration
can leave background preparation competing with the next indexed pass. Full
1,000-scene epoch measurements are preserved separately to isolate iteration.
This increases how far ahead shards may be prepared/downloaded; it does not
cache decoded scenes. Users with tight remote cache limits can override it.

Validation: MDS roundtrip/custom encoding/multi-stream tests pass; explicit
prefetch override tested. No storage-format change.

Full epochs confirm the gain: native .9494 → .4025, Torch 1.0010 → .5105,
PyG 1.0709 → .6099 ms/scene (five passes over all 1,000 records).

### Reuse per-shard column decoders — `9f3ce59`

Profiling indexed reads showed 4,500 codec-dispatch calls per 300 records,
including 3,000 ndarray schema parses. A shard subclass caches the immutable
standard int/ndarray decoders and uses buffer offsets for size headers. Other
encodings retain Mosaic dispatch. Only exact standard MDS shard instances are
replaced, after safety validation; shard I/O and streaming scheduling stay
upstream. No format change or decoded-record cache.

Random indexed medians, native / Torch / PyG (ms/scene):

- Previous commit: .1882 / .2502 / .3502.
- Planned decoders: .1469 / .2096 / .2976.
- Independent repeat: .1425 / .2047 / .3021.
- Reduction versus previous repeat: 24.3% / 18.2% / 13.7%.

Iteration repeat: .4044 / .5113 / .6224 ms/scene. Benefit there is smaller
because Mosaic thread coordination still dominates. See `mds-codecs*.json`.
Tests compare raw columns with Mosaic for dynamic/static dtypes and shapes,
custom strings/bytes/JSON, compression, pickling, list/array/slice indexing and
iteration. All 1,000 real records also match source and upstream decoding.

### Rejected: memoryview within ndarray decoding

Passing per-column memoryviews to ndarray decoding avoids its internal second
payload copy, without retaining the entire row for a single small field.
However native random reads only changed .1425 → .1379 ms, while Torch and PyG
changed .2047 → .2103 and .3021 → .3024 ms. Repeated reads and iteration were
also inconsistent. Reverted: too little end-to-end benefit to justify the
buffer-type compatibility assumption. Raw results: `mds-array-buffer.json`.

### DataLoader benchmark extension — `1db2b30`

`--batch-size 8 --workers 0` additionally measures map-style adapter loading
and batching. Torch uses lists because agent/map sizes vary; PyG uses the
library's time-padding collator and builds a `Batch`. `--workers 2` uses spawn
and includes process startup on every pass (no persistent workers). These
timings should not be mistaken for steady-state training throughput. Native
streaming iteration remains measured separately. Worker smoke runs cover
both full and forecast adapters, including shard decoder serialization.

### Parse each row's variable-size header once — `469e6fa`

The post-codec profile still showed one NumPy scalar parse per variable column.
Parse all sizes together with `struct.unpack_from`, using a format string
planned per shard, and consume those sizes directly instead of allocating a
second list. Native-endian uint32 layout matches Mosaic. A first prototype
stored a `Struct` object; serialization tests rejected that because it cannot
be pickled. The retained implementation stores only a string and integer size.

Indexed-only random latency versus `916235e`, native / Torch / PyG:

- Before: .1410 / .2002 / .2972 ms.
- After: .1346 / .2009 / .2897 ms.
- Repeat: .1321 / .1921 / .2888 ms.

Native improved 4.6–6.3% and PyG 2.5–2.8%; Torch's smaller gain is less stable
(−0.4% to 4.0%). Retained because the native gain is reproducible and the change
removes per-row work with little added complexity. Additional tests cover
all-fixed-size schemas and empty byte columns. All 1,500 real scenes still
match upstream values/dtypes/shapes exactly. See `mds-header*.json`.

## Cross-backend results after codec caching

These measurements precede the final size-header optimization; final cumulative
measurements for that progression are recorded below.

Waymo: five passes of 300 scenes; median milliseconds per scene. Percentages
are latency reductions against the original `090b74d` benchmark baseline.

| Path | Random before → after | Reduction | Iteration before → after | Reduction |
|---|---|---:|---|---:|
| Native | .1817 → .1462 | 19.6% | 1.0065 → .3935 | 60.9% |
| Torch | .2537 → .2029 | 20.0% | 1.0361 → .5054 | 51.2% |
| PyG | .3417 → .3053 | 10.7% | 1.0977 → .6264 | 42.9% |

Random min–max across final passes: native .1426–.1556, Torch .2008–.2113,
PyG .2993–.3357 ms. Repeated reads: .1151 / .1707 / .2668 ms, improvements
of 24.8% / 20.5% / 12.0%. Mixed-pattern native sequential reads were .1989
versus .1840 ms (8.1% slower), with a wide .1467–.2212 range; background work
from interrupted epochs affects the next pass. Torch/PyG sequential medians
improved to .2082/.3127 ms.

An isolated indexed-only run (no interrupted streaming epochs) resolved this
ambiguity: native sequential .1833 → .1414 ms (22.9% faster), Torch
.2539 → .2047 (19.4%), PyG .3429 → .2952 (13.9%). Random reads in that run:
.1874 → .1410, .2542 → .2002, .3518 → .2972 ms. The mixed-pattern slowdown is
not present when indexing runs alone. Raw files: `mds-*-indexed.json`.

Complete 1,000-scene epochs, measured separately: native .9494 → .3513 ms
(63.0% reduction), Torch 1.0010 → .4470 (55.3%), PyG 1.0709 → .5560 (48.1%).
Forecast random reads: native .2181 → .1700, Torch .2750 → .2294, PyG
.3810 → .3361 ms. Forecast iteration: 1.0421 → .4893, 1.0086 → .5464,
1.1436 → .6601 ms. See `mds-final-epoch.json` and `mds-final-forecast.json`.

Final random reads for the same source through other backends (native / Torch
/ PyG): pickle .0717/.1308/.2140 ms; Zarr 4.7418/4.8600/4.9778 ms. Pickle
remains faster locally. These are page-cache-warm read comparisons, not
claims about disk or remote-storage throughput.

Batch size eight, zero-worker DataLoader: MDS Torch .2266 ms/scene; PyG
.6956 ms/scene including time padding and graph batching. Conversion-only
cost was .0458 ms for Torch and .1124 ms for PyG. PyG batching/conversion is
therefore a substantial remaining downstream cost.

Opening MDS now took 3.54–3.75 ms across adapter cases versus 1.99–2.80 ms at
baseline: decoder planning adds startup work. First reads were .32/.73/.82 ms
versus .33/.77/.68 ms. Each is a single observation per isolated process,
so initialization/first-read differences are less certain than pass timings.

### Independent dataset: ETH pedestrians

Created 500 scenes from real source data in `$TRAJ_DATA/eth`, preserving the
raw files. These have no maps and 77 post-resampling frames. Sample feature
payloads span 6,468–45,276 bytes (median 21,560), versus Waymo's
5,096–685,412 (median 140,140). Reproduction:

```sh
.venv/bin/prejectory process eth --input "$TRAJ_DATA/eth" \
  --output data/eth-mds --storage-backend mds --read native --read-split test \
  --assign none --limit 500 --jobs 1 --no-progress --force
.venv/bin/python examples/repack_dataset.py data/eth-mds data/eth-pickle --backend pickle
.venv/bin/python benchmark_io.py data/eth-mds data/eth-pickle \
  --count 500 --repeats 5 --batch-size 8 --output eth-results.json
```

The original implementation was run from a detached `090b74d` worktree via
`PYTHONPATH`, against the identical new MDS export. Five full-epoch passes:

| Path | Random before → after | Reduction | Iteration before → after | Reduction |
|---|---|---:|---|---:|
| Native | .1571 → .1124 | 28.5% | .9059 → .3038 | 66.5% |
| Torch | .2092 → .1656 | 20.9% | .9574 → .4326 | 54.8% |
| PyG | .3005 → .2591 | 13.8% | 1.0206 → .5153 | 49.5% |

### Tradeoffs and limits

- No MDS format or writer changes were needed; existing exports work as-is.
- Disk usage is unchanged by the reader optimizations. Generated Waymo MDS
  occupies 225 MiB, pickle 227 MiB, compressed Zarr 58 MiB; ETH MDS 12 MiB and
  pickle 14 MiB (filesystem allocated sizes). These reflect different formats
  and codecs, not a size improvement from this work.
- Decoder caching stores only small per-column objects per shard, not decoded
  payloads. Higher predownload may prepare/download more shards in advance;
  explicit user limits remain configurable. Peak RSS was not measured.
- Benchmarks cover two datasets on one host and local uncompressed MDS. No
  OS-cache eviction, network latency, distributed training, GPU transfer, or
  compressed throughput claims. Compressed shard correctness is tested.
- Spawned worker runs are correctness/smoke checks and include startup cost;
  they are not evidence for steady-state multi-worker speedups.
- Future opportunities: Mosaic preparation/ready-thread coordination; shard
  offset/file-access costs; PyG graph collation; persistent-worker training
  benchmarks. File-handle/mmap caching would require careful remote eviction
  and worker-lifecycle handling, so it was not added speculatively.

## Final cumulative results (including `469e6fa`)

All values below are median **milliseconds per scene**, five passes. Random
Waymo values use the original mixed-pattern workload (300 samples). Epoch
values use all 1,000 Waymo / 500 ETH scenes. These are the final production code.

| Dataset/path | Random baseline → final | Reduction | Full epoch baseline → final | Reduction |
|---|---|---:|---|---:|
| Waymo native | .1817 → .1386 | 23.7% | .9494 → .3549 | 62.6% |
| Waymo Torch | .2537 → .1989 | 21.6% | 1.0010 → .4472 | 55.3% |
| Waymo PyG | .3417 → .2914 | 14.7% | 1.0709 → .5485 | 48.8% |
| ETH native | .1571 → .1103 | 29.8% | .9059 → .2949 | 67.4% |
| ETH Torch | .2092 → .1550 | 25.9% | .9574 → .3830 | 60.0% |
| ETH PyG | .3005 → .2545 | 15.3% | 1.0206 → .5175 | 49.3% |

Waymo sequential/random/repeated/iteration in the mixed-pattern run:

- Native: .1805 / .1386 / .1083 / .4112.
- Torch: .2066 / .1989 / .1626 / .5140.
- PyG: .3006 / .2914 / .2570 / .6005.

Indexed-only confirmation: sequential .1288/.1925/.2908 and random
.1321/.1921/.2888 ms for native/Torch/PyG. Interrupted-epoch background work
still affects mixed runs, so do not substitute these lower numbers into the
mixed-workload comparison.

Final forecast random reads: .1732/.2244/.3224 ms; forecast iteration:
.4911/.5273/.6417 ms. Batch size eight, no DataLoader workers:
Torch full/forecast .2207/.2386; PyG full/forecast .6928/.7527 ms per scene.

Raw final measurements: `mds-header-pipeline.json`, `mds-header-epoch.json`,
`mds-header-eth.json`; isolated indexed repetitions: `mds-header.json` and
`mds-header-repeat.json`. Full distributions are retained in each file.
The final worker smoke run is in `mds-header-workers.json`. The final profile
is `mds-header-profile.txt`; header NumPy calls are gone and remaining time
is spread across ndarray decoding, record validation and file I/O. Further
changes are deferred because likely gains are smaller or require more invasive
storage/streaming changes.

## Validation

- Full non-slow suite: 268 tests passed.
- Full basedpyright: zero errors and warnings; changed Python files pass Ruff.
- All 1,000 Waymo and 500 ETH records compared against the upstream Mosaic
  decoder and a second representation with exact array values, dtypes and
  shapes; iteration order also matched.
- Compressed/custom encoding and shard serialization tests pass; full and
  forecast Torch/PyG DataLoader paths work with two spawned workers.
- Profiles (`results/*profile.txt`) are diagnostic, not latency
  benchmark: cProfile adds overhead. Remaining costs include ndarray decoding,
  scene validation and shard file reads, rather than repeated schema parsing.
