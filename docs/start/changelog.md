# Changelog

## Unreleased — Python API simplification

- Remove the Lyft Level 5 integration and its optional dependencies because the dataset is no
  longer publicly available.
- Add `prejectory.plan`, `run`, and manifest-aware `open_dataset`.
- Replace request `config_path` with `config` (TOML path or `ProjectConfig`). Replace
  `RuntimeOverride` with `DatasetConfigPatch`; CLI flag conversion is internal.
- Simplify `ProjectConfig.resolve_dataset_config` to accept a dataset name or descriptor.
- Add request task selection: omitted inherits, a name/inline task replaces, `None` disables.
- Rename plan `data_root` to `input_dir` and `resolved_config` to `config`; expose absolute
  paths, a summary, and diagnostics. Compiled descriptor/loader/assignment fields are private.
  Use `plan.config.runtime` and `plan.include_map` for execution settings and map availability.
- Rename `SceneRecord.mask` and Torch/PyG `agent_time_mask` to `valid_mask`.
  Replace `split(origin, end)` with `forecast(PredictionBounds(origin, end))` and rename
  `SplitSceneRecord`/`TorchSplitSceneRecord` to `ForecastRecord`/`TorchForecastRecord`.
- Plan prediction bounds now use `PredictionBounds` rather than a tuple.
- Export configuration models from `prejectory.config`, descriptor split support from
  `prejectory.datasets`, counters/progress from `prejectory.runtime`, reader contracts from
  `prejectory.io`, and exceptions from `prejectory.errors`. Remove dataset configuration aliases
  and the mutable public `TRAJECTORY_SCHEMAS` registry.
- Python execution is quiet by default, with optional Rich rendering and progress callbacks.
  `ExecutionResult.written_scenes` forwards the canonical `stats` counter.
- Stage output before publication; failed overwrites preserve the previous export.
- Manifest format is now **3**, with payload ID/version and available split counts.
  Regenerate existing exports: older manifests and renamed pickle record layouts are not
  compatible. MDS retains the physical `mask` column while exposing `valid_mask` in Python.
- Custom `OutputTransform` requires `format_id` and a transform; readers require an explicit
  decoder. Custom MDS output also requires columns, validated during planning.

## v2.0.0 - June 2026

Major redesign of `prejectory`: move the project from a clonable research/template
repository with preprocessing scripts, models, metrics, and training entry points into
an installable trajectory data processing library with a shared runtime, typed
configuration, dataset registry, and documented Python and CLI surfaces.

### Breaking changes

- Replace the old top-level `preprocessing`, `datamodules`, `models`, `metrics`, and
  training-script layout with the package-based `src/prejectory` library structure.
- Remove the bundled model-development and evaluation stack from the package scope so
  `prejectory` now focuses on dataset ingestion, scene construction, map handling, export,
  and reading processed data.
- Replace dataset-specific preprocessing scripts and YAML config files with registered
  dataset integrations, shared loader interfaces, and TOML-based runtime configuration.
- Redesign configuration around explicit `runtime`, `scenes`, `screening`, `map`,
  `read`, `assign`, `output`, and `dataset` sections. Input selection and output split
  assignment are now separate concepts.
- Change the default persisted output path to the backend-neutral scene-record model and
  make `pickle` the default storage backend.

### New

- Add the public runtime request/plan API: `ExecutionRequest`, `resolve_request()`,
  `execute_request()`.
- Add a Typer-based CLI with dataset inspection, config display, split-support checks,
  processing, dry-run planning, progress reporting, `--version`, and external dataset
  registration through `--dataset-module`.
- Add a dataset registry built around `DatasetDescriptor`, with built-in integrations for the
  supported datasets and a documented path for custom dataset loaders.
- Add shared scene, schema, category, split, and map domain models under `prejectory.core`.
- Add a shared processing pipeline for scene windowing, resampling, split assignment,
  lane-change sampling, screening, map extraction, and output writing.
- Add a storage and reading layer under `prejectory.io`, including manifests,
  backend-neutral `SceneRecord` data, `pickle`, `mds`, and `null` backends, readers, and
  optional Torch/PyG adapters.
- Add layered TOML configuration with reusable profiles, dataset-local overrides, and
  focused runtime overrides from the CLI and Python API.
- Add a new documentation site structure covering concepts, getting started guides,
  configuration reference, dataset reference pages, output formats, and API reference.
- Add a focused automated test suite for the public API, configuration, runtime, CLI,
  IO roundtrips, adapters, processing pipeline, screening, registry behavior, and dataset
  integration contracts.

### Changes

- Rework map processing into reusable map graph builders, parsers, compilers, and
  deferred scene-level map resolution.
- Rework dataset implementations to use shared loader and resource abstractions instead
  of one-off preprocessing entry points.
- Update packaging metadata, optional extras, CI, and release workflow for the library
  distribution model.

## v1.3.0 - August 2025

- Expand dataset support across Argoverse 1 and 2, Waymo, nuScenes, Lyft, VoD, ApolloScape, NGSIM, ETH/UCY, openDD, and AD4CHE.
- Update dependency and container setup for the broader dataset surface.
- Refresh documentation structure.

## v1.2.0 - March 2025

- Add the INTERACTION dataset.
- Extend lane-graph attributes and visualization tools.
- Update requirements and container recipes.

## v1.1.1 - December 2024

- Fix minor preprocessing issues in the lane-change sampling.
- Ship small documentation updates.

## v1.1.0 - September 2024

- Add exiD, uniD, SIND, and A43.
- Publish a pre-built Docker image and PyPI installation path.

## v1.0.0 - April 2024

- Publish initial public release.
