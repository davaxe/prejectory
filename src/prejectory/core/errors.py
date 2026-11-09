"""Project-specific exception hierarchy used across Prejectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from prejectory.core.categories import DatasetSplit


class PrejectoryError(Exception):
    """Base class for all exceptions raised by Prejectory."""


@dataclass(slots=True)
class CliError(Exception):
    """A formatted CLI error with an explicit process exit code."""

    message: str
    exit_code: int = 1

    @override
    def __str__(self) -> str:
        """Return the error message for display."""
        return self.message


def cli_usage_error(message: str) -> CliError:
    """Create a CLI error representing invalid user input."""
    return CliError(message=message, exit_code=2)


class ConfigurationError(ValueError, PrejectoryError):
    """Raised when user-provided configuration is invalid or incomplete."""


class LoaderConfigError(ConfigurationError, PrejectoryError):
    """Raised when there is an issue with a loader configuration."""


class TrajectorySchemaError(ConfigurationError, PrejectoryError):
    """Raised when a trajectory schema is invalid, unknown, or cannot be applied."""


class MissingOptionalDependencyError(PrejectoryError):
    """Raised when an optional dependency required by a feature is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        dependencies: tuple[str, ...] = (),
        install_target: str | None = None,
    ) -> None:
        super().__init__(message)
        self.dependencies: tuple[str, ...] = dependencies
        self.install_target: str | None = install_target


class DatasetRegistryError(ValueError, PrejectoryError):
    """Raised when dataset registry metadata is invalid or inconsistent."""


class DatasetNotFoundError(PrejectoryError):
    """Raised when a dataset name cannot be resolved from the registry."""

    def __init__(self, dataset_name: str, available_datasets: list[str]) -> None:
        known = ", ".join(available_datasets) or "none"
        super().__init__(f"Unknown dataset '{dataset_name}'. Available datasets: {known}.")
        self.dataset_name: str = dataset_name
        self.available_datasets: tuple[str, ...] = tuple(available_datasets)


class SplitError(ValueError, PrejectoryError):
    """Base class for dataset split-related errors."""


class SplitNotSupportedError(SplitError):
    """Raised when a dataset does not support the requested split.

    This error is raised when one or more specific splits are requested from a
    dataset or loader that does not provide them.

    Parameters
    ----------
    loader_name : str
        The name of the loader or dataset that does not support the split.
    split : DatasetSplit or str or list[DatasetSplit | str]
        The requested split or splits.

    """

    def __init__(
        self,
        loader_name: str,
        split: DatasetSplit | str | list[DatasetSplit] | list[str],
    ) -> None:
        def _display(value: DatasetSplit | str) -> str:
            return value.value if isinstance(value, DatasetSplit) else str(value)

        requested = split if isinstance(split, list) else [split]
        rendered = [_display(item) for item in requested]
        super().__init__(f"{loader_name} does not support split '{', '.join(rendered)}'.")
        self.loader_name: str = loader_name
        self.split: list[str] = rendered


class SplitAssignmentError(SplitError):
    """Raised when runtime split-assignment data is missing or invalid."""


class ManifestCompatibilityError(ValueError, PrejectoryError):
    """Raised when a persisted dataset manifest is incompatible with this version."""

    def __init__(self, format_version: int, supported_version: int) -> None:
        msg = (
            f"Unsupported manifest format version '{format_version}'. "
            f"Supported version: {supported_version}."
        )
        super().__init__(msg)
        self.format_version: int = format_version
        self.supported_version: int = supported_version


class MissingPredictionBoundsError(ValueError, PrejectoryError):
    """Raised when a forecast view is requested for a task-free record."""

    def __init__(self, scene_number: int) -> None:
        super().__init__(
            f"Scene {scene_number} has no prediction bounds; provide an explicit override.",
        )
        self.scene_number: int = scene_number


class UnsupportedStorageBackendError(ValueError, PrejectoryError):
    """Raised when an unknown storage backend is requested."""

    def __init__(self, storage_backend: str, supported_backends: tuple[str, ...]) -> None:
        supported = ", ".join(supported_backends)
        super().__init__(
            f"Unsupported storage backend '{storage_backend}'. Supported backends: {supported}.",
        )
        self.storage_backend: str = storage_backend
        self.supported_backends: tuple[str, ...] = supported_backends
