# Zarr trajectory read optimization

Branch: `perf/zarr-read-optimization`. Base: `1a60be3`.

The benchmark uses the existing 1,000-scene Waymo export at `data/zarr`,
derived from the Waymo data available under `$TRAJ_DATA`. It contains six
shards, float32 trajectory features with 91 frames and seven features per
frame, and varying agent and map sizes. The original export is unchanged.
`benchmark_zarr.py` times complete `SceneRecord` loads with a fixed random
sample (seed 12345), five repetitions for each 100-scene progression, and
fresh reader construction each repetition. All values below are median
milliseconds per scene for the first sequential and randomized passes. The
committed JSON files under `benchmarks/results/` contain every sample,
minimum, maximum, repeated passes, opening time, and payload signatures.

| Commit | Hypothesis and retained change | Sequential | Random | Random change vs prior |
| --- | --- | ---: | ---: | ---: |
| `1a60be3` | Original reader baseline | 12.113 | 12.051 | — |
| `4e16a5f` | Add a repeatable end-to-end benchmark; reader unchanged | 12.113 | 12.051 | — |
| `cf17a93` | Cache small scene index and offset columns per shard | 10.927 | 10.837 | 10.1% faster |
| `d466e1e` | Reuse the last decoded trajectory feature chunk | 4.988 | 9.185 | 15.2% faster |
| `4341805` | Reuse the last decoded companion and map chunks | 0.693 | 8.435 | 8.2% faster |
| `3e06a3f` | Retain two feature chunks for randomized reuse | 0.682 | 6.824 | 19.1% faster |
| `e6fd61c` | Retain two companion agent chunks | 0.690 | 6.010 | 11.9% faster |
| `f03f8ce` | Retain eight recent map chunks | 0.690 | 5.503 | 8.4% faster |
| `c4af720` | Evict old shard readers when estimated cache capacity exceeds 256 MiB | 0.699 | 5.606 | Within run variation |
| `9b0c768` | Reject invalid slice bounds without stalling | 0.702 | 5.641 | Within run variation |

Profiling the original reader on 50 scenes found 500 Zarr array accesses and
about 508 chunk decodes. A 100-scene field timing put trajectory features at
587 ms, with other arrays contributing roughly 510 ms. Repeated decoding of
large feature chunks, followed by repeated access to smaller agent and map
chunks, dominated complete-record latency. After caching, a random 100-scene
pass still spent about 260 ms decoding 37 missed feature chunks and about
250 ms across map arrays. This guided the cache capacities above.

For a separate cross-shard comparison, the original reader at `1a60be3` and
the final reader used the same benchmark, 250 scenes, seed 12345, three
repetitions. Sequential first-pass median fell from **11.816 to 0.676 ms**
per scene (94.3% lower); randomized first-pass median fell from **11.822 to
5.182 ms** (56.2% lower). Repeated-pass medians changed from 11.764 to
0.563 ms sequentially and 11.711 to 4.891 ms randomly. The final reader
over all 1,000 scenes, three repetitions, measured 0.617 ms sequentially
and 4.647 ms randomly on the first pass; repeated-pass medians were 0.494
and 4.676 ms. Reader construction was about 5 to 6 ms and is reported
separately from scene loads.

An unsuccessful layout experiment rewrote a separate copy of this export
with 256-agent and 2,048-map-element chunks, leaving the original intact.
All fields, shapes, and dtypes matched on 39 sampled records, but at 100
scenes it increased sequential median latency from 0.693 to 3.353 ms and
left random median latency effectively unchanged (8.435 to 8.367 ms). Disk use rose from 56,727,805
to 59,201,752 bytes (4.4%). The layout and temporary conversion script
were discarded. No retained storage-format change or source-data rewrite
was needed.

The cache held about 140 MiB of decoded payloads after the 100-scene random
sample across six shards: 117 MiB features, 5 MiB companion agent arrays,
and 18 MiB map arrays. The new 256 MiB bound applies to the estimated
capacity of cached shard readers. A single shard whose chunks exceed the
bound can still be retained so a read succeeds. Returning NumPy copies
keeps records independent even while decoded chunks are shared internally.

Correctness checks included the Zarr writer/reader tests, tests for slices
crossing agent and map chunk boundaries, record mutation isolation, shard
cache eviction, and invalid slice bounds, plus field-by-field hashes and
dtype/shape comparisons for 39 Waymo records against the original reader.
Ruff checks and all tests outside `tests/test_datasets.py` passed. The
full suite was attempted but the parallel raw-dataset tests failed because
the existing `AssertingSceneWriter` helper lacks `flush_local()`, which
the executor calls. The non-slow suite had the same unrelated failure in
its four-job mocked-registry case.

The first pass is only cold-ish: the benchmark does not clear the OS page
cache, and later repetitions benefit from it. The benchmark has one derived
Waymo export and a single process; other datasets, storage devices, many
worker processes, and genuinely cold storage may shift the best cache size.
The reader API materializes complete records, so partial trajectory slices
are not separately timed. Remaining work would measure a broader set of
exports and consider a bounded cache shared across workers or a scene-packed
format if randomized reads need further improvement.
