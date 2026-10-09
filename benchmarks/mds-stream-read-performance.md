# MDS reads across many streams

Base: `7ec966b`; branch: `codex/mds-stream-read-performance`.

`examples/benchmark_mds_streams.py` combines actual processed exports through
Mosaic `Stream` objects. The baseline uses all 19 training exports from
`/home/davax/dev/wayfinder/source/data/processed`, with no reprocessing or format
changes. Each stream contributes 256 samples per complete, shuffled epoch.
This keeps dataset sizes from dominating the mixture. Random indexing also
samples 256 seeded indices per dataset, spanning the full stored exports.
The indexed wrapper exposes the stored sample count because Mosaic's epoch
length can be smaller than its index space when `Stream(choose=...)` is used.

Each stream-count/adapter case runs in a fresh subprocess. Five timed passes
follow one warmup for each workload. The benchmark checks scene counts and
dataset contributions and records identity checksums for comparing versions.
Torch uses one intra-op thread and the adapters' default copy behavior. OS page
caches are not flushed. These results measure local reads, not network or cold
disk throughput. RSS includes imports and framework memory; Python/platform
and dependency versions are recorded in each JSON file.

Reproduce from the checkout:

```sh
PYTHONPATH=src .venv/bin/python examples/benchmark_mds_streams.py \
  /home/davax/dev/wayfinder/source/data/processed \
  --per-stream 256 --output benchmarks/results/mds-streams-baseline.json
```

Use `--streams all --view forecast` to measure forecast conversions.
`--workers 0` adds iterable DataLoader batching; positive worker counts use
spawn and persistent workers, with worker startup in the excluded warmup.
Torch batches preserve lists; PyG uses the existing time-padding collator.
Loader consumption reads batch metadata without unbatching PyG graphs.

## Baseline

Median milliseconds per scene; random indexing / shuffled iteration:

| Streams | Native | Torch | PyG |
|---|---|---|---|
| 1 | .1130 / .3038 | .1654 / .4009 | .2445 / .5056 |
| 4 | .1226 / .3839 | .1594 / .4126 | .2589 / .5458 |
| 19 | .1201 / .4083 | .1655 / .4526 | .2618 / .5373 |

Raw distributions and identity checksums: `results/mds-streams-baseline.json`.
The host is an Intel Xeon E-2144G with four cores/eight hardware threads.

## Progression 1: synchronous iteration of resident local shards

The baseline profile showed preparation calls and file-lock acquisition during
every local epoch. The reader now skips preparation threads and readiness
polling when all shards are resident, uncompressed MDS files, every stream has
no remote, and no cache limit is set. It retains Mosaic's world detection,
epoch counter/resumption, weighting, shuffle, partitioning and sample reads.
Starting another epoch invalidates an interrupted iterator. Other cases use
Mosaic's original iterator. `local_iteration=False` disables the shortcut.

19-stream iteration medians, native / Torch / PyG (ms/scene):

- Baseline: .4083 / .4526 / .5373.
- Local iteration: .1308 / .1991 / .2889.
- Latency reduction: 68.0% / 56.0% / 46.2%.

Random indexing in this run was .1227 / .1721 / .2829, versus baseline
.1201 / .1655 / .2618. The indexed code path has not changed; these measured
increases (2.2% / 4.0% / 8.0%) need final repeated comparisons before concluding
there is no regression. Raw samples: `results/mds-streams-local-iteration.json`.
All identity checksums, counts and dataset histograms match the baseline exactly.

Validation compares three epochs to Mosaic with shuffle on/off, weighting with
under/oversampling, and batch padding. Tests cover interrupted epochs, Mosaic
checkpoint resumption, fallback for compression/remotes/cache limits and an
explicit opt-out, plus both adapters using two spawned persistent workers.
No format changes or decoded-record caches are introduced.

## Progression 2: integer and ndarray decoding

Standard integers now decode directly from native-endian signed int64 bytes.
Fixed-dtype ndarray decoders cache NumPy dtypes, parse packed dimensions into
Python tuples, and use a buffer offset for the payload. This removes NumPy
shape-array allocation and a second payload copy. Each view retains only its
own column's bytes (plus its small shape header), preserving read-only output.
Dynamic dtypes and custom encodings retain Mosaic's decoding behavior.

19-stream medians, native / Torch / PyG (ms/scene):

- Random: .1068 / .1624 / .2561.
- Iteration: .1252 / .1716 / .2695.

Raw samples: `results/mds-streams-decoders.json`. Tests cover every supported
fixed value dtype, all four legal shape widths, signed int64 boundaries,
fixed/dynamic shapes and dtypes, custom codecs, compression and pickling.

## Progression 3: cache immutable shard offsets

The shard reader caches its sample-offset table lazily, using four bytes per
entry. Each subsequent access opens the raw file, seeks directly to its row,
reads the payload and closes the file. No file descriptors or decoded scenes
are retained, and a missing file still triggers Mosaic's preparation/retry.
Offset bytes survive spawn serialization and remain valid after an immutable
shard is evicted and downloaded again. A remote download/eviction/re-read test
and compressed/pickled decoder tests cover these cases.

19-stream medians, native / Torch / PyG (ms/scene):

- Random: .1007 / .1530 / .2417 (5.7% / 5.8% / 5.6% below progression 2).
- Iteration: .1113 / .1619 / .2514.

Raw samples: `results/mds-streams-offsets.json`. All 2,432 sampled real scenes
tested after progression 2 matched Mosaic's columns, dtypes, shapes and
writability. Final verification repeats this after all retained changes.

## Progression 4: remove temporary tensor clones during PyG collation

The time-padding collators now shallow-copy PyG attribute stores before
assigning padded tensors. Batch construction concatenates into independent
output tensors, so cloning all agent/map tensors beforehand is redundant.
Tests for full and forecast views verify every input tensor is unchanged,
including after all output tensors are mutated. Existing time alignment and
padding checks remain intact.

With 19 streams, 128 samples each, three passes, batch size eight and zero
workers, PyG loader latency changed .6235 → .5482 ms/scene (12.1% lower) versus
the immediately preceding MDS progression. The original baseline was .9403.
Files: `results/mds-streams-offsets-pyg-loader.json` and
`results/mds-streams-pyg-collation.json`. Final comparisons include forecasts
and both adapters with spawned persistent workers.

## Final comparison and independent repetitions

The final production changes are in `9f05f9d`, `b101397`, `e3ff757` and
`a0f8a22`. The original source was preserved without switching branches:

```sh
mkdir -p .cache/mds-stream-base
git archive 7ec966b src/prejectory | tar -x -C .cache/mds-stream-base
PYTHONPATH=.cache/mds-stream-base/src .venv/bin/python examples/benchmark_mds_streams.py \
  /home/davax/dev/wayfinder/source/data/processed --streams all --per-stream 256 \
  --output benchmarks/results/mds-streams-baseline-repeat.json
PYTHONPATH=src .venv/bin/python examples/benchmark_mds_streams.py \
  /home/davax/dev/wayfinder/source/data/processed --per-stream 256 \
  --output benchmarks/results/mds-streams-final-repeat.json
```

The JSON records the imported reader's source path. Both baselines and both
final runs are retained. The table below uses the independent repeat baseline
and the final repeat: 19 streams, 4,864 scenes/epoch, five passes, ms/scene.

| Path | Random before → after | Reduction | Iteration before → after | Reduction |
|---|---|---:|---|---:|
| Native | .1219 → .0997 | 18.2% | .3367 → .1141 | 66.1% |
| Torch | .1712 → .1610 | 5.9% | .4418 → .1739 | 60.6% |
| PyG | .2658 → .2436 | 8.3% | .5565 → .2608 | 53.1% |

The first final run gave iteration .1099 / .1641 / .2583 and random
.0964 / .1526 / .2455. Relative to the original baseline, iteration latency
fell 73.1% / 63.7% / 51.9%. Native baseline iteration varied between runs,
so the repeated comparison above provides a more conservative assessment.

Scaling remains stable: final iteration for 1 / 4 / 19 streams was
.1060 / .1188 / .1141 native, .1602 / .1608 / .1739 Torch, and
.2541 / .2542 / .2608 PyG. Initial single-stream PyG random reads were
.2445, versus .2487/.2542 in the two final runs; their pass ranges overlap.
The repeated original-code measurement was .2869, so that small initial
median increase was not reproduced as a consistent regression. Raw small
mixture repetitions are preserved rather than treating a single timing as
proof of a gain in every workload.

Full/forecast loader results use 19 streams, 128 scenes per stream, three
passes and batches of eight. Startup is excluded by warmup:

| Workload | Torch before → after | PyG before → after |
|---|---|---|
| Full, zero workers | .5334 → .1783 | .9403 → .5640 |
| Forecast, zero workers | .4668 → .2023 | 1.0751 → .6509 |
| Full, two persistent workers | 1.6739 → 1.6859 | 1.9660 → 1.7624 |

Zero-worker full loading improves 66.6% for Torch and 40.0% for PyG.
Forecast iteration is .3613 → .1368 native, .4641 → .1897 Torch, and
.6405 → .2954 PyG, reductions of 62.1%, 59.1% and 53.9%. Forecast indexed
reads also improve for all three paths. Torch's two-worker loader is roughly
unchanged (+0.7% latency); PyG improves 10.4%. Both versions load more slowly
with two workers than without them on this host. Process communication and
tensor transfer remain substantial costs; these measurements do not establish
benefits from increasing worker count.

Every paired workload matches record counts, identity checksums and dataset
histograms, including all worker/forecast epochs. Machine-readable comparisons
are in `results/mds-streams-comparison.json`.

## Validation and limits

- 340 tests passed, with 46 raw-dataset/slow integration cases deselected,
  matching the CI unit-test selection. Ruff, formatting checks and Basedpyright
  pass with zero errors/warnings. The tensor ownership tests were also rerun
  after final typing annotations.
- 2,432 real scenes across all 19 datasets match Mosaic's independently read
  raw bytes and decoded columns exactly, including dtypes, shapes, scalar
  types and writability. `results/mds-streams-validation.json` records this.
- The exports contain 1,215,994 stored training scenes. Caching every shard's
  offset table would use 4,870,496 bytes (4.65 MiB) per reader/worker; the
  correctness sample populated 2,316,056 bytes. There is no scene-payload or
  file-descriptor cache. RSS measurements cover each parent process, including
  framework imports, and do not sum worker memory.
- Planning filenames/dtypes and checking local residency add startup work.
  Most 19-stream opens were about 99–108 ms in final runs versus 81–90 ms in
  baseline runs, with outliers in both versions. Throughput gains apply after
  opening. Existing exports remain compatible and require no reprocessing.
- The shortcut calls Mosaic's private world/epoch helpers, tested with
  Streaming 0.13.0. `local_iteration=False` retains the upstream iterator.
  Remote, compressed and cache-limited correctness is tested; their throughput,
  distributed training, cold disk reads and GPU transfer are not measured.

Final artifacts: `mds-streams-final.json`, `mds-streams-final-repeat.json`,
`mds-streams-final-loader.json`, `mds-streams-final-forecast.json` and
`mds-streams-final-workers.json`, all under `results/`. Full timing samples
and the repeat baselines are retained alongside them.
