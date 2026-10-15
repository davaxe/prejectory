# Outputs and schemas

Output decisions in `prejectory` are easiest to understand as two separate choices: what each scene
contains, and how those scenes are stored on disk.

For exact configuration keys, see the [output reference](../reference/configuration/output.md).

## Mental model

There are two layers:

1. the **trajectory schema**, which decides which per-timestep features are present
2. the **storage backend**, which decides how encoded scenes are persisted

That separation is deliberate. The same logical scene can be written through different backends.

## Trajectory schema

The schema controls the feature tensor, not the storage format.

Built-in schemas are:

| Schema name                       | Feature fields                          | Feature dim |
| --------------------------------- | --------------------------------------- | ----------- |
| `positions_only`                  | `x`, `y`                                | 2           |
| `positions_yaw`                   | `x`, `y`, `yaw`                         | 3           |
| `positions_velocity`              | `x`, `y`, `vx`, `vy`                    | 4           |
| `positions_velocity_yaw`          | `x`, `y`, `vx`, `vy`, `yaw`             | 5           |
| `positions_velocity_acceleration` | `x`, `y`, `vx`, `vy`, `ax`, `ay`        | 6           |
| `canonical`                       | `x`, `y`, `vx`, `vy`, `ax`, `ay`, `yaw` | 7           |

If the output schema requests fields that are not present in the dataset's native schema, the scene
is converted during scene building and the manifest records which features were derived.

## Precision and recentering

These settings apply independently of backend:

- `precision` controls numeric dtype for persisted floating-point arrays (`float32` or `float64`)
- `recenter_positions` subtracts one per-scene `(x, y)` offset before writing and stores that
  offset in each scene record

`float32` is typically preferred for storage and training throughput, while `float64` can be useful
for precision-sensitive workflows.

## Storage backends

The built-in backends are:

| Backend  | Purpose                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------- |
| `pickle` | Writes one pickled `SceneRecord` per scene. No extra dependency.                                |
| `mds`    | Writes Mosaic Streaming shards and `index.json` files. Requires `prejectory[mds]`.               |
| `zarr`   | Writes chunked worker-local Zarr shards with indexed scene access. Requires `prejectory[zarr]`.  |
| `null`   | Runs the full pipeline but does not persist scene data. Useful for validation and benchmarking. |

Two defaults are easy to miss:

- the CLI `process` command defaults to `pickle`
- the Python `ExecutionRequest` model also defaults to `pickle`

If you want consistent behavior across both entry points, choose the backend explicitly.

## Output layout

The runtime writes one root `manifest.json` plus split directories.

Scene data is then written under split directories:

- `train`, `val`, and `test` when a split strategy produces those partitions
- `unsplit` when no split strategy is active

The backend decides what goes inside each split directory:

- `pickle` writes `*.pkl` files
- `mds` writes shard files plus `index.json`
- `zarr` writes `part-*.zarr` worker shards
- `null` writes no scene files

The manifest is written as JSON at the output root. It records the produced splits, storage backend,
feature columns, trajectory schema fields, numeric precision, horizon, optional prediction task,
and map availability for the run.

## Maps in persisted records

Persisted scene records always include map arrays.

- when map data is available, these arrays contain node and edge data
- when map data is unavailable or disabled, readers expose empty map arrays

Semantic labels and encoded arrays use different names:

- `agent_category` is the semantic DataFrame column before record encoding
- `agent_ids` preserves the source identifier corresponding to each tensor row
- `agent_types` is the encoded per-agent integer array in `SceneRecord`
- `EdgeType` and map node types are semantic map labels during map construction
- `map_edge_types` and `map_node_types` are encoded arrays in `SceneRecord`

## Readers and adapters

All backends feed the same reader-side mental model:

- `PickleReader`, `MDSReader`, and `ZarrReader` yield framework-neutral `SceneRecord` objects
- `TorchSceneDataset` and `HeteroSceneDataset` build on top of readers.
