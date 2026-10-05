"""Persisted metadata models shared across storage backends."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from prejectory.core.errors import ManifestCompatibilityError

FORMAT_VERSION: int = 3
MANIFEST_FILENAME: str = "manifest.json"
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PredictionTaskManifest:
    """Configured and effective prediction bounds for one processed export."""

    name: str | None
    """Descriptor-owned task name, or `None` for a custom task."""
    source_prediction_origin: int
    source_prediction_end: int
    prediction_origin: int
    prediction_end: int


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Format-agnostic metadata stored alongside exported datasets.

    The manifest records the shape and schema contract of one processed export.
    Reader and adapter code use it to understand feature columns, temporal
    horizons, coordinate handling, map availability, and manifest compatibility.
    For custom payload formats, trajectory metadata describes the requested
    canonical representation; the format ID/version owns the actual payload layout.
    """

    dataset: str
    """Name of the DatasetSource dataset, e.g. 'nuscenes' or 'waymo'."""
    storage_backend: str
    """Storage backend used to write the exported scene records."""
    prejectory_version: str
    """Package version that produced the manifest."""
    source_trajectory_schema: str
    """Trajectory schema emitted by the dataset loader before conversion."""
    source_trajectory_schema_fields: tuple[str, ...]
    """Semantic field names emitted by the dataset loader before conversion."""
    trajectory_schema: str
    """Trajectory schema stored in the exported records."""
    trajectory_schema_fields: tuple[str, ...]
    """Semantic field names stored in exported trajectory records."""
    derived_features: tuple[str, ...]
    """Output features derived during schema conversion."""
    feature_columns: tuple[str, ...]
    """Per-timestep feature columns stored in record tensors."""
    horizon_frames: int
    """Number of full-horizon frames per persisted record."""
    precision: str
    """Floating-point precision used for exported feature arrays."""
    recenter_positions: bool
    """Whether spatial values were recentered around each scene."""
    has_map: bool
    """Whether records may contain map topology arrays."""
    sample_time: float
    """Output `sample_time` interval in seconds after resampling."""
    original_sample_time: float
    """Dataset `sample_time` interval in seconds before resampling."""
    prediction_task: PredictionTaskManifest | None = None
    """Prediction task selected for this export, if any."""
    format_version: int = FORMAT_VERSION
    """Manifest schema version used for compatibility checks."""
    dataset_names: tuple[str, ...] = ()
    """Dataset names indexed by the integer dataset ids stored in records."""

    payload_format: str = "prejectory.scene"
    """Record encoding identifier; other formats require an explicit decoder."""
    payload_version: int = 1
    """Version of the payload format, independent of the manifest schema."""
    split_counts: dict[str, int] = field(default_factory=dict)
    """Available output partitions and their committed record counts."""

    @property
    def splits(self) -> tuple[str, ...]:
        """Available partitions in their persisted iteration order."""
        return tuple(self.split_counts)

    def __post_init__(self) -> None:
        """Validate temporal manifest fields."""
        if not self.payload_format.strip() or self.payload_version < 1:
            msg = "Payload format must be named and its version positive."
            raise ValueError(msg)
        if any(name not in {"unsplit", "train", "val", "test"} for name in self.split_counts):
            msg = "Manifest contains an unknown output split."
            raise ValueError(msg)
        if any(type(count) is not int or count < 0 for count in self.split_counts.values()):
            msg = "Manifest split counts must be non-negative integers."
            raise ValueError(msg)
        if not self.dataset_names:
            object.__setattr__(self, "dataset_names", (self.dataset,))
        if self.horizon_frames <= 0:
            msg = f"`horizon_frames` must be positive, but got {self.horizon_frames}."
            raise ValueError(msg)
        task = self.prediction_task
        if task is not None:
            if not 0 < task.source_prediction_origin < task.source_prediction_end:
                msg = "Invalid source prediction bounds in manifest."
                raise ValueError(msg)
            if not 0 < task.prediction_origin < task.prediction_end <= self.horizon_frames:
                msg = "Invalid effective prediction bounds in manifest."
                raise ValueError(msg)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        """Create a manifest from previously serialized JSON data."""
        format_version = int(payload.get("format_version", 0))
        if format_version != FORMAT_VERSION:
            raise ManifestCompatibilityError(format_version, FORMAT_VERSION)
        return cls(
            dataset=str(payload["dataset"]),
            format_version=format_version,
            storage_backend=str(payload["storage_backend"]),
            payload_format=str(payload["payload_format"]),
            payload_version=int(payload["payload_version"]),
            split_counts=dict(payload["split_counts"]),
            prejectory_version=str(payload["prejectory_version"]),
            source_trajectory_schema=str(
                payload.get("source_trajectory_schema", payload["trajectory_schema"]),
            ),
            source_trajectory_schema_fields=tuple(payload["source_trajectory_schema_fields"]),
            trajectory_schema=str(payload["trajectory_schema"]),
            trajectory_schema_fields=tuple(payload["trajectory_schema_fields"]),
            derived_features=tuple(payload.get("derived_features", ())),
            feature_columns=tuple(payload["feature_columns"]),
            horizon_frames=int(payload["horizon_frames"]),
            precision=str(payload["precision"]),
            recenter_positions=bool(payload["recenter_positions"]),
            has_map=bool(payload["has_map"]),
            sample_time=float(payload["sample_time"]),
            original_sample_time=float(payload["original_sample_time"]),
            prediction_task=_parse_prediction_task(payload.get("prediction_task")),
            dataset_names=tuple(payload.get("dataset_names", (payload["dataset"],))),
        )


def _parse_prediction_task(payload: object) -> PredictionTaskManifest | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        msg = "`prediction_task` must be an object or null."
        raise TypeError(msg)
    task_payload = cast("dict[str, Any]", payload)
    return PredictionTaskManifest(
        name=(None if task_payload.get("name") is None else str(task_payload["name"])),
        source_prediction_origin=int(task_payload["source_prediction_origin"]),
        source_prediction_end=int(task_payload["source_prediction_end"]),
        prediction_origin=int(task_payload["prediction_origin"]),
        prediction_end=int(task_payload["prediction_end"]),
    )


def package_version() -> str:
    """Return the installed prejectory package version for manifest metadata."""
    try:
        return version("prejectory")
    except PackageNotFoundError:
        return "0+unknown"


def manifest_path(root: str | Path) -> Path:
    """Return the manifest path for one storage root.

    Parameters
    ----------
    root: str | Path
        The root directory of the processed dataset.

    Returns
    -------
    Path
        The path to the manifest file.
    """
    return Path(root) / MANIFEST_FILENAME


def write_manifest(root: str | Path, manifest: DatasetManifest) -> None:
    """Write the storage manifest for one output root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing manifest", extra={"root": str(root), "format_version": FORMAT_VERSION})
    _ = manifest_path(root).write_text(
        json.dumps(manifest.to_json_dict(), indent=2),
        encoding="utf-8",
    )


def read_manifest(root: str | Path) -> DatasetManifest:
    """Read and parse the storage manifest for one output root.

    Parameters
    ----------
    root: str | Path
        The root directory of the processed dataset.

    Returns
    -------
    DatasetManifest
        The parsed dataset manifest.
    """
    logger.debug("Reading manifest", extra={"root": str(root)})
    payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return DatasetManifest.from_json_dict(payload)
