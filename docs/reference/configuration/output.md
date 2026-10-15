# `[output]` section

Output settings control what is persisted after preprocessing. In practice, this means choosing the trajectory schema, numeric precision, and any backend-specific output options.

| Key | Type | Description | Default |
|---|---|---|---|
| `trajectory_schema` | `str` or `table` | Output trajectory schema to persist. Use either a built-in trajectory schema name or a structured custom trajectory schema definition. | `"canonical"` |
| `precision` | `"float32"` or `"float64"` | Floating-point precision of persisted data. | `"float32"` |
| `recenter_positions` | `bool` | Offset all agent positions by the scene mean before writing. The offset is stored in the output. | `true` |

## `[output.mds]` section

This nested block only matters when writing MDS output and is used to tune shard writing behavior.

| Key | Type | Description | Default |
|---|---|---|---|
| `compression` | `str` | Compression setting for MDS shards, for example `"zstd:7"`. | `none` |
| `hashes` | `array[str]` | Hash algorithms to apply to MDS shard files. | `none` |
| `size_limit` | `int` or `str` | Shard size limit, in bytes or as the backend-supported size string. | `67_108_864` |
| `exist_ok` | `bool` | Overwrite existing shard files when set to `true`. | `false` |

!!! note "MDS parameters"
    This section can be used to override MDS-specific parameters for the MDS
    storage backend. These values are only used when the selected storage
    backend is `mds`.

## `[output.zarr]` section

Zarr stores each worker's output as an independent shard and uses these values
to tune array chunks and compression.

| Key | Type | Description | Default |
|---|---|---|---|
| `scene_chunk` | `int` | Chunk length for scene metadata and offset arrays. | `256` |
| `agent_chunk` | `int` | Chunk length for flattened agent arrays. | `4096` |
| `map_node_chunk` | `int` | Chunk length for flattened map-node arrays. | `16384` |
| `map_edge_chunk` | `int` | Chunk length for flattened map-edge arrays. | `16384` |
| `compression_level` | `int` or `none` | Zstandard compression level from 0 to 22; `none` disables compression. | `3` |

## Minimal example

```toml
[datasets.a43.output]
trajectory_schema = "positions_velocity_yaw"
precision = "float64"
recenter_positions = true

[datasets.a43.output.mds]
compression = "zstd:7"
hashes = ["sha1", "xxh64"]
size_limit = 33554432
exist_ok = true

[datasets.a43.output.zarr]
scene_chunk = 128
agent_chunk = 2048
compression_level = 5
```

The canonical concept is the trajectory schema, and the TOML key is
`output.trajectory_schema`.

To use a custom trajectory schema instead of a built-in one, define it with a
table. Custom trajectory schemas must include the base fields
`frame`, `id`, `agent_category`, `x`, and `y`:

```toml
[datasets.a43.output]
trajectory_schema = { name = "custom", fields = ["frame", "id", "agent_category", "x", "y", "vx", "vy"] }
precision = "float64"
recenter_positions = true
```
