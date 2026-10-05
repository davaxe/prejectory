# First run (CLI)

The current CLI exposes five commands: `available`, `inspect`, `show-config`, `split-support`, and
`process`. A good first run is: inspect the dataset, preview the resolved config, then run
`process --plan` before writing anything.

## Inspect the registry

```bash
prejectory available
prejectory available --no-details
prejectory inspect a43
prejectory split-support a43
```

`available` lists the datasets that are usable in the current environment. `inspect` shows a
dataset's defaults, native schema, source-sequence bounds before windowing, read and assignment
support, feature support, and loader options.
`split-support` shows which read and assignment strategies are available for that dataset.
Use `available --no-details` for a compact registry listing.

!!! note "`available` and `inspect` reflect the current environment"

    The dataset registry only exposes entries whose optional dependencies are currently available.
    If a dataset is missing, check the installation instructions and make sure you have the right
    extras installed.
    
    It does **not** validate that the dataset's expected input files are actually present. If the dataset is listed but you get an error when you try to process, check that the input path is correct and contains the expected files.

## Preview the resolved config

```bash
prejectory show-config a43 --config config.toml
```

`show-config` resolves the same dataset defaults, profile fragments, dataset entry, and CLI
overrides that `process` uses, but it stops before execution. Use it when you want to confirm the
effective `scenes`, `screening`, `read`, `assign`, `map`, `output`, and `loader_options` settings.

`show-config` and `process` both default to the `pickle` backend. Pass
`--storage-backend` when you want to preview a different backend.

## Plan a processing run

=== ":material-linux: Bash"

    ```bash
    prejectory process a43 \
        --input data/a43/raw \
        --output data/a43/processed \
        --plan
    ```

=== ":fontawesome-brands-windows: PowerShell"

    ```ps1
    prejectory process a43 `
        --input data/a43/raw `
        --output data/a43/processed `
        --plan
    ```

`--plan` resolves the full run and prints a summary without executing it. The summary includes the
dataset, paths, backend, worker count, schema, map usage, read strategy, assignment strategy, and
any loader options that affect the run.

Add `--config config.toml` when you have a project config file. Add read or assignment overrides
only after checking `prejectory split-support <dataset>`; for example, `a43` does not support
`preserve-native` because it has no native partitions.

The `process` command currently defaults to the `pickle` backend. Choose a different backend
explicitly when needed:

- `pickle` for the simplest persisted output with no extra dependency
- `mds` for Mosaic Streaming shards, which requires `prejectory[mds]`
- `zarr` for chunked, indexed arrays, which requires `prejectory[zarr]`
- `null` to execute the pipeline without writing any dataset files

## Useful options

| Option | Effect |
| --- | --- |
| `--storage-backend` | Choose a registered storage backend such as `pickle`, `mds`, `zarr`, or `null`. |
| `--scene-schema` | Override the output trajectory schema for this run. |
| `--jobs` | Override worker count. Values above `1` enable parallel execution. |
| `--progress/--no-progress` | Enable or disable the live progress display during processing. |
| `--limit` | Stop after producing the requested number of scenes. |
| `--seed` | Set the random seed for assignment and other randomized operations. |
| `--read` | Choose the input read mode, currently `all` or `native`. |
| `--read-split` | Select native dataset partitions when `--read native` is used. |
| `--assign` | Choose the output assignment mode such as `preserve-native`, `scene`, `source`, `time`, or `shuffled-time`. |
| `--ratio`, `--gap`, `--segments` | Tune assignment behavior for compatible assignment modes. |
| `--include-map/--no-map` | Force map inclusion on or off for datasets that support maps. |
| `--overwrite/--no-overwrite` | Explicitly replace a non-empty output directory. Existing output is rejected by default. |

## Execute the run

When the plan looks right, rerun the same command without `--plan`. If `--force` is omitted, the
CLI prints the summary again and asks for confirmation before processing starts.

The output directory must be empty unless `--overwrite` is supplied. Input and
output directories must not overlap.

After a successful run, check the output directory:

- `manifest.json` at the dataset root describes the produced dataset
- assignment subdirectories such as `train`, `val`, `test`, or `unsplit` contain the backend-specific
  data files
