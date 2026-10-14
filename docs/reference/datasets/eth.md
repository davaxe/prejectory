# ETH

ETH is one of the classic pedestrian trajectory benchmarks used in socially aware forecasting. It is usually discussed together with the UCY scenes and is valued for crowd interaction studies rather than map-rich vehicle forecasting.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2009</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 20 frames (benchmark origin 8) @ 2.5 Hz |
| Effective horizon | 77 frames (benchmark origin 29) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 89-1440 frames (observed) |
| Resampling | 4:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Keep agents with at least 2 observations |
| Maps | None |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | ETH BIWI pedestrian scene layout |
| Loader expectation | The loader uses the ETH/UCY raw text layout and does not parse a dataset-specific version marker. |

## Normalization

### Agent categories

| Dataset type | Prejectory type |
| ------------ | -------------- |
| Any tracked actor | `PEDESTRIAN` |

### Map types

| Dataset type | Prejectory type |
| ------------ | -------------- |
| Not applicable | Not applicable |

## Split support

Use the CLI for current native split and assignment support.

```bash
prejectory split-support eth
```

## References

- Dataset paper: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Benchmark mode (default)

The names `eth`, `hotel`, `univ`, `zara1`, and `zara2` identify benchmark folds
by default. The loader reads tab-separated `frame id x y` files from `train/`,
`val/`, and `test/`, normalizing frame numbers independently per file.

A fold's training data can contain other pedestrian domains. For example,
`eth/train` can contain Hotel, University, and Zara recordings. Different
benchmark-fold exports can therefore share observations.

To preserve the supplied benchmark labels:

```bash
prejectory process eth --input data/ethucy/eth --output processed/eth-benchmark-new \
  --assign preserve-native --seed 42 --force
```

Without `--assign`, the existing default remains unassigned output (`unsplit`).

## Sequence mode (config file only)

Sequence mode selects actual pedestrian domains and disregards benchmark split
labels. Save this as `eth-ucy-sequence.toml`, or use the repository's
`examples/eth_ucy_sequence.toml`:

```toml
[defaults.loader_options]
input_mode = "sequence"

[defaults.read]
strategy = "all"

[defaults.assign]
strategy = "time"
ratio = { train = 0.8, val = 0.1, test = 0.1 }
gap = 0
```

For a dataset-specific setting, put `input_mode` under
`[datasets.eth.loader_options]` instead. There is no `--input-mode` CLI flag.
Omitting the setting keeps benchmark mode.

```bash
for domain in eth hotel univ zara1 zara2; do
  prejectory process "$domain" --config eth-ucy-sequence.toml --input data/ethucy \
    --output "processed/${domain}-sequence-new" --seed 42 --force
done
```

The input can contain one benchmark fold, several folds, or whole recordings.
The loader recursively groups files by recording name. The mapping is explicit
and case sensitive; optional `_train`, `_val`, and `_test` suffixes identify
fragments of the same recording.

| Domain | Required filename stems |
| ------ | ----------------------- |
| `eth` | `biwi_eth` |
| `hotel` | `biwi_hotel` |
| `univ` | `students001`, `students003` (both required) |
| `zara1` | `crowds_zara01` |
| `zara2` | `crowds_zara02` |

`uni_examples` and `crowds_zara03` are explicitly excluded from every sequence
domain. Benchmark mode continues to read them as supplied.

Only the established four-column `frame id x y` text format is supported,
with coordinates in metres and ten original frame units per 0.4-second
annotation timestep. Tabs and spaces are accepted. Convert native `obsmat` or
UCY pixel/`.vsp` annotations to this format first. Unknown `.txt` filenames,
missing required recordings, malformed observations, and incompatible frame
units raise errors.

Whole recordings such as `originals/biwi_eth.txt` are preferred when found
outside `train/`, `val/`, and `test/` folders. Otherwise all matching copies
and fragments are merged. Deduplication uses recording identity, original
frame number, and pedestrian ID. Different coordinates for the same observation
raise an error.

Normalization occurs once after merging. It subtracts the recording's earliest
frame and divides by ten, preserving missing time intervals. Fragments must
retain their original timestamps and coordinate system; independently
renumbered or recentered fragments cannot be reliably reconstructed.

Each recording remains a separate processing source. University windows never
span its two recordings. To keep their pedestrian IDs distinct, University uses
`2 * original_id` for `students001` and `2 * original_id + 1` for `students003`.
IDs in single-recording domains and all dataset IDs remain unchanged.

Sequence mode requires `read = all`, chronological `assign = time`, and strict
windowing. The existing processing pipeline partitions each recording before
windowing and resampling, so complete scene horizons stay within one split and
partitions share no observations. `gap` optionally removes additional source
timesteps between partitions. Short partitions or screening may produce no
output windows; inspect the standard manifest's split counts.

Ratios control timeline spans rather than output counts. Chronological splits
are deterministic; the seed does not change their boundaries. Existing tasks,
schemas, screening, and resampling still apply. Use `scenes.resample` to change
the output sampling rate.

These commands use new output directories and omit `--overwrite`. `--force`
only skips the interactive prompt; it does not replace existing nonempty
exports. `--plan` inspects configuration without writing data.
