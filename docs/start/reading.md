# Reading data

`prejectory` readers expose one common in-memory record model across storage backends. This means
you can switch between `pickle`, `mds`, and `zarr` without changing your downstream scene-processing code.

## Reader model

Framework-neutral readers return [`SceneRecord`](../reference/api/io/index.md#prejectory.io.SceneRecord)
objects with:

- scene id and position offset
- full-horizon agent features and masks
- optional map graph arrays

Use [`SceneRecord.forecast`](../reference/api/io/index.md#prejectory.io.SceneRecord.forecast)
when a model needs explicit observation and prediction tensors.

Use the same post-processing logic regardless of backend.

## Open a dataset

`open_dataset()` reads the manifest, selects the backend, checks partitions and record counts,
and returns records together with their metadata. It accepts string and `Path` inputs.

<!-- no-validate -->
```python
from prejectory import open_dataset

dataset = open_dataset("output")  # All available partitions, in manifest order.
print(dataset.manifest.feature_columns)
print(dataset.manifest.split_counts)
record = dataset[0]
print(record.features.shape, record.valid_mask.shape)
forecast = record.forecast()  # Uses the record's prediction task.
```

Omit `split` to read all partitions, or use `split="train"`, `"val"`, `"test"`, or `"unsplit"`.
Missing paths and unknown partitions raise errors; an existing empty partition has length zero.
Both iteration and integer indexing (including negative indices) are supported. MDS iteration
preserves backend worker partitioning. Use the backend-specific readers below for remote streams
or advanced backend settings; their default `split=None` continues to mean `unsplit/`.

Forecast views share their arrays with the original record. To override bounds, use
`record.forecast(PredictionBounds(origin, end))`, with half-open bounds in stored frames.
Task-free records require explicit bounds. NumPy and Torch records both use `valid_mask`;
forecast records use `history_mask` and `future_mask`.

Only open trusted pickle exports: unpickling can execute code.

## Read from Zarr output

Install the Zarr extra before using the reader: `pip install prejectory[zarr]`.

<!-- no-validate -->
```python
from pathlib import Path
from prejectory.io.readers import ZarrReader

reader = ZarrReader(Path("output"), split="train")
scene = reader[0]
print(scene.features.shape, scene.map_edge_indices.shape)
```

The reader combines all worker shards in the selected split and supports
integer and negative indexing.

## Read the manifest first

<!-- no-validate -->
```python
from pathlib import Path
from prejectory.io import read_manifest

manifest = read_manifest(Path("output"))
print(manifest.feature_columns)
print(manifest.trajectory_schema_fields)
print(manifest.horizon_frames, manifest.prediction_task)
```

Reading the manifest up front with
[`read_manifest()`](../reference/api/io/index.md#prejectory.io.read_manifest) is the easiest way to verify
schema, horizon, and precision.

!!! tip "Readable manifest"
    The manifest is stored in a human-readable JSON format, so you can also
    open and inspect it manually if needed.


## Read from Pickle output

!!! danger "Only read trusted pickle files"
    Python pickle deserialization can execute arbitrary code. Do not use
    `PickleReader` with files from untrusted or unverifiable sources.

<!-- no-validate -->
```python
from pathlib import Path
from prejectory.io.readers import PickleReader

reader = PickleReader(Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.features.shape, scene.valid_mask.shape)

if scene.prediction_bounds is not None:
    split = scene.forecast()
    print(split.history_features.shape, split.future_features.shape)
```

For unsplit exports, use `split=None` (the default), which reads from `unsplit/`.

`SceneRecord.agent_ids` maps each agent tensor row back to its source identifier.
Built-in datasets also provide `dataset_id`; custom registered datasets may use
`dataset_id = None`, in which case the manifest's `dataset` field is authoritative.

## Read from MDS output

!!! warning "MDS requires extra dependencies"
    Install the MDS extra before using MDS readers: `pip install prejectory[mds]`.

<!-- no-validate -->
```python
from pathlib import Path
from prejectory.io.readers import MDSReader

reader = MDSReader(path=Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.features.shape, scene.valid_mask.shape)
```

For unsplit exports, use `split=None` (default), which reads from `unsplit/`.

Mosaic streams can be combined directly. Prediction bounds travel with each row, so shuffling does
not lose the task boundary:

<!-- no-validate -->
```python
from streaming import Stream
from prejectory.io.readers import MDSReader

reader = MDSReader(
    streams=[
        Stream(local="processed/argoverse1", split="train"),
        Stream(local="processed/eth", split="train"),
    ],
    batch_size=32,
    shuffle=True,
)

sample = next(iter(reader)).forecast()
```

## Torch and PyG adapters

On top of the readers, `prejectory` provides optional adapters:

- [`TorchSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.TorchSceneDataset) for full-horizon
  Torch tensor records
- [`HeteroSceneDataset`](../reference/api/io/adapters.md#prejectory.io.adapters.HeteroSceneDataset) for full-horizon
  PyTorch Geometric `HeteroData`
- `TorchForecastDataset` and `HeteroForecastDataset` for task-aware history/future views

Use these when your training stack expects framework-native dataset objects.

Forecast adapters use the bounds stored in each row, including when Mosaic combines and shuffles
streams with different task definitions. Explicit bounds replace row metadata when training all
streams with one common task:

<!-- no-validate -->
```python
from prejectory.io import PredictionBounds
from prejectory.io.adapters import HeteroForecastDataset

dataset = HeteroForecastDataset(reader)
common_task = HeteroForecastDataset(reader, bounds=PredictionBounds(20, 50))
```

## Choosing a reader setup

- Use `open_dataset()` for ordinary local exports with manifest validation.

- Use [`PickleReader`](../reference/api/io/readers.md#prejectory.io.readers.PickleReader) for simple local
  workflows and easy inspection.
- Use [`MDSReader`](../reference/api/io/readers.md#prejectory.io.readers.MDSReader) for larger-scale or
  streaming-oriented training pipelines.
- Keep reader-side code backend-neutral by depending on the shared
  [`SceneRecord`](../reference/api/io/index.md#prejectory.io.SceneRecord) contract.

## Custom payloads

Custom persisted output must declare a format ID and version. The manifest's trajectory metadata
describes the canonical input to a record transform; the payload's layout is owned by its format.
A scene transform also owns conversion, recentering, precision, and schema semantics.

```python
from prejectory.io import SceneRecord
from prejectory.runtime import OutputTransform


def scene_number(record: SceneRecord) -> dict[str, int]:
    return {"number": record.scene_number}


output = OutputTransform(
    record_transform=scene_number,
    format_id="example.scene-number",
    format_version=1,
)
```

Pass `output_transform=output` on the request. For MDS, also provide
`mds_columns={"number": "int"}`. Planning validates this requirement and backend dependencies.
Multiprocess runs require transforms defined as importable top-level callables.

<!-- no-validate -->
```python
from prejectory import open_dataset
from prejectory.io import read_manifest

manifest = read_manifest("output")
assert (manifest.payload_format, manifest.payload_version) == ("example.scene-number", 1)
numbers = open_dataset("output", decoder=lambda payload: int(payload["number"]))
```

Custom formats always require an explicit decoder. It receives the unpickled payload for pickle,
or the raw row for MDS. No decoder registry or implicit imports are used.
