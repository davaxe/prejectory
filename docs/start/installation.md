# Installation

The base `prejectory` package is intentionally small. Install it first, then add optional extras for the CLI, MDS storage, Torch/PyG adapters, or dataset-specific loaders when needed. The `viz` extra is currently reserved for future visualization support.

## Requirements

- Python `>=3.10`
- `pip` or `uv`

## Install the base package

```bash
pip install prejectory
```

The base package includes:

- the dataset registry and runtime planning API
- the processing pipeline
- built-in scene and schema types
- the `pickle` storage backend
- the framework-neutral [`PickleReader`](../reference/api/io/readers.md#prejectory.io.readers.PickleReader)

It does **not** include the CLI, MDS backend, Torch adapters, PyTorch Geometric adapters, any visualization implementation, or dataset-specific optional dependencies.

## Install optional features

Use extras to enable additional functionality.

```bash
pip install "prejectory[cli]"
pip install "prejectory[cli,mds]"
pip install "prejectory[torch]"
pip install "prejectory[pyg]"
```

You can combine extras in one command:

```bash
pip install "prejectory[cli,mds,torch,pyg]"
```

## Available extras

| Extra | Adds |
| --- | --- |
| `cli` | Typer/Rich command-line interface |
| `mds` | MDS writer backend and [`MDSReader`](../reference/api/io/readers.md#prejectory.io.readers.MDSReader) |
| `torch` | Torch dataset adapters such as [`TorchSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.TorchSceneDataset) |
| `pyg` | PyTorch Geometric adapters such as [`HeteroSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.HeteroSceneDataset) |
| `viz` | Reserved placeholder extra for future visualization support. It currently adds no dependencies. |
| `waymo` | Optional dependencies for the Waymo dataset |
| `ad4che` | Optional dependencies for the AD4CHE dataset |

## Recommended installs

| Use case | Command |
| --- | --- |
| Basic processing with pickle output | `pip install prejectory` |
| CLI workflows | `pip install "prejectory[cli]"` |
| MDS output and reading | `pip install "prejectory[cli,mds]"` |
| Torch training pipelines | `pip install "prejectory[torch]"` |
| PyG graph pipelines | `pip install "prejectory[pyg]"` |

## Readers and adapters

| API | Required extra |
| --- | --- |
| [`PickleReader`](../reference/api/io/readers.md#prejectory.io.readers.PickleReader) | none |
| [`MDSReader`](../reference/api/io/readers.md#prejectory.io.readers.MDSReader) | `mds` |
| [`TorchSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.TorchSceneDataset) | `torch` |
| [`HeteroSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.HeteroSceneDataset) | `pyg` |

## Dataset-specific dependencies

Most built-in datasets are available with the base install. Some datasets require additional optional dependencies.

| Dataset | Required extra |
| --- | --- |
| `waymo` | `waymo` |
| `ad4che` | `ad4che` |

The dataset registry only exposes datasets whose dependencies are installed. As a result, `prejectory available` reflects the datasets supported by your current environment.

For example:

```bash
pip install "prejectory[cli,waymo]"
prejectory available
```

## Verify the installation

For the base package:

```bash
python -c "import prejectory"
```

For the CLI:

```bash
prejectory available
prejectory inspect a43
```

For the MDS reader:

```bash
python -c "from prejectory.io.readers import MDSReader"
```

For Torch adapters:

```bash
python -c "from prejectory.io.adapters import TorchSceneDataset"
```

For full-horizon PyG adapters:

```bash
python -c "from prejectory.io.adapters import HeteroSceneDataset"
```
