"""Persisted storage contracts and export-facing configuration.

## Import guide

``python
from prejectory.io import (
    DatasetManifest,
    SceneRecord,
    ForecastRecord,
    read_manifest,
    write_manifest,
)
``

Output configuration models live under [`prejectory.config`][] so the
runtime and CLI share one canonical configuration surface.

## Related modules

- [`prejectory.io.adapters`][] for optional Torch and PyG adapter layers
"""

from prejectory.io.base import DatasetReader, IterableDatasetReader, StorageBackend
from prejectory.io.dataset import OpenedDataset, open_dataset
from prejectory.io.manifest import (
    DatasetManifest,
    PredictionTaskManifest,
    manifest_path,
    read_manifest,
    write_manifest,
)
from prejectory.io.records import ForecastRecord, PredictionBounds, SceneRecord

__all__ = [
    "DatasetManifest",
    "DatasetReader",
    "ForecastRecord",
    "IterableDatasetReader",
    "OpenedDataset",
    "PredictionBounds",
    "PredictionTaskManifest",
    "SceneRecord",
    "StorageBackend",
    "manifest_path",
    "open_dataset",
    "read_manifest",
    "write_manifest",
]
