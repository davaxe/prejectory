"""Writer backends used by the runtime.

This package is the extension point for persisted output backends. The runtime
selects a backend by string name and resolves it to a worker-local writer
provider through :func:`build_writer_provider`.

Built-in backends:

- `pickle` writes one pickled `SceneRecord` per scene
- `mds` writes Mosaic Streaming shards
- `zarr` writes chunked worker-local array shards
- `null` executes the full pipeline but discards scene data

Custom backends can be added by registering another backend builder with
`register_writer_backend`.
"""

from prejectory.io.backends.provider import build_writer_provider

__all__ = ["build_writer_provider"]
