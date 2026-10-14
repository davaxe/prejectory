# Univ

Univ is a pedestrian-only scene from the ETH/UCY benchmark family. It is commonly used in socially aware human trajectory forecasting, especially in comparisons that report results scene by scene.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2007</strong></div>
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
| Source bounds | 148-1440 frames (observed) |
| Resampling | 4:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Keep agents with at least 2 observations |
| Maps | None |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | UCY UNIV scene layout |
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
prejectory split-support univ
```

## References

- Family reference: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
univ/
├── train/
├── val/
└── test/
```

## Sequence domain exports

The default reads benchmark folds. To export only this pedestrian domain, select
sequence mode in a config file. See the shared [ETH/UCY modes, filename mapping,
reconstruction rules, and examples](eth.md#sequence-mode-config-file-only).
