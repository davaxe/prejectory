"""Explicit dataset registry and descriptor models."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from prejectory.config.models import PredictionTaskConfig
from prejectory.core.errors import (
    DatasetNotFoundError,
    DatasetRegistryError,
    LoaderConfigError,
    MissingOptionalDependencyError,
)
from prejectory.processing.loading.models import LoaderOptionsModel, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from prejectory.config.models import DatasetConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.core.scene import TrajectorySchema
    from prejectory.datasets.shared.resources import MapProviderFactory
    from prejectory.processing.loading.base import SceneLoader
    from prejectory.processing.maps import MapProvider
    from prejectory.processing.models import LoaderPlan


_REGISTRY: dict[str, DatasetDescriptor] = {}
logger = logging.getLogger(__name__)

SourceTemporalUnit = Literal["recording", "scenario", "case", "scene", "batch", "unknown"]
"""Logical sequence unit represented by one source before scene windowing."""

FrameBoundConfidence = Literal["exact", "documented", "observed", "estimated", "unknown"]
"""Confidence level for descriptor-reported source-frame bounds."""

WindowPolicyName = Literal["strict", "anchored", "partial"]
"""Supported sliding-window completeness policies."""

WindowValidationMode = Literal["error", "warn", "off"]
"""How request resolution should handle windows outside descriptor bounds."""


@dataclass(slots=True, frozen=True, kw_only=True)
class DatasetSplitSupport:
    """Assignment strategies supported by a dataset integration.

    These flags describe how the runtime may create train/validation/test
    partitions when `assign.strategy` is not `none` or `preserve-native`.
    Datasets should only enable strategies whose required grouping information
    is stable and available from their sources.
    """

    scene: bool = True
    """Whether scenes may be assigned independently by scene id."""
    source: bool = False
    """Whether all scenes from the same source may be assigned together."""
    time_block: bool = False
    """Whether source timelines may be divided into contiguous split blocks."""


@dataclass(slots=True, frozen=True, kw_only=True)
class DatasetFeatureSupport:
    """Optional dataset features supported by a dataset integration."""

    map: bool = False
    """Whether the dataset can provide map data when map inclusion is enabled."""
    lane_change_sampling: bool = False
    """Whether lane-change-aware sampling is supported through a `lane_id` signal."""


@dataclass(slots=True, frozen=True, kw_only=True)
class FrameBounds:
    """Frame-count bounds for the source sequence before windowing.

    Bounds describe the logical unit emitted by the loader before generic
    scene-window extraction. Use `None` when a bound is not known without
    scanning the dataset.
    """

    min_frames: int | None = None
    """Smallest known source sequence length, in source frames."""
    max_frames: int | None = None
    """Largest known source sequence length, in source frames."""
    varying: bool = False
    """Whether source sequence lengths vary within the dataset."""
    confidence: FrameBoundConfidence = "unknown"
    """How reliable the reported bounds are."""


@dataclass(slots=True, frozen=True, kw_only=True)
class DatasetWindowingSupport:
    """Dataset-level support metadata for generic sliding-window extraction."""

    enabled_by_default: bool = False
    """Whether the built-in dataset defaults enable scene windowing."""
    default_policy: WindowPolicyName = "strict"
    """Completeness policy expected by the dataset defaults."""
    supported_policies: tuple[WindowPolicyName, ...] = ("strict",)
    """Windowing policies that are meaningful for this dataset."""
    max_window_frames: int | None = None
    """Largest supported requested window length, when statically known."""
    validation: WindowValidationMode = "error"
    """How strictly runtime planning validates known window bounds."""


@dataclass(slots=True, frozen=True, kw_only=True)
class DatasetTemporalSupport:
    """Minimal temporal metadata for reasoning about dataset windowing."""

    source_unit: SourceTemporalUnit = "unknown"
    """Logical source unit being windowed, such as a recording or scenario."""
    source_frame_bounds: FrameBounds = field(default_factory=FrameBounds)
    """Frame-count bounds for the source unit before windowing."""
    windowing: DatasetWindowingSupport = field(default_factory=DatasetWindowingSupport)
    """Supported generic windowing behavior for this dataset."""


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Descriptor for one dataset integration.

    A `DatasetDescriptor` is the registry object that connects a dataset key, such as
    `"a43"` or `"waymo"`, to the loader and defaults needed by the runtime. The
    CLI, config resolver, and Python runtime all resolve dataset names to this
    object before planning a run.

    The descriptor owns dataset-level metadata rather than per-run state. It defines
    how to build loaders, which trajectory fields the raw loader produces, what
    configuration should be used as the starting point, which split strategies
    are valid, and how run-scoped map providers are initialized.

    Parameters
    ----------
    name : str
        Unique registry key used in config files, CLI commands, and
        `ExecutionRequest.dataset`.
    loader_cls : type[SceneLoader[Any, Any]]
        Loader class implementing the dataset integration.
    default_config : DatasetConfig
        Dataset-specific baseline configuration. User profiles, dataset entries,
        and runtime overrides are applied on top of this value.
    native_schema : TrajectorySchema
        Trajectory schema emitted by the loader before output schema conversion.
    supported_native_splits : tuple[DatasetSplit, ...] or None, optional
        Dataset-provided partitions available to `read.strategy = "native"` and
        `assign.strategy = "preserve-native"`. Use `None` for datasets without
        native partitions.
    loader_options_model : type[LoaderOptionsModel], optional
        Typed model for dataset-owned options under `[datasets.<name>.loader_options]`.
    map_provider_factory : MapProviderFactory or None, optional
        Optional context-manager factory that prepares a run-scoped map
        provider.
    feature_support : DatasetFeatureSupport, optional
        Optional dataset features supported by this integration.
    split_support : DatasetSplitSupport, optional
        Custom assignment modes supported by this dataset in addition to native
        split preservation.
    temporal_support : DatasetTemporalSupport or None, optional
        Optional metadata describing source sequence lengths and generic
        windowing semantics.
    tasks : Mapping[str, PredictionTaskConfig], optional
        Named prediction tasks supported by this dataset.
    default_task : str or None, optional
        Named task selected when a request does not make an explicit choice.
    """

    name: str
    loader_cls: type[SceneLoader[Any, Any]]
    default_config: DatasetConfig
    native_schema: TrajectorySchema
    supported_native_splits: tuple[DatasetSplit, ...] | None = None
    loader_options_model: type[LoaderOptionsModel] = NoLoaderOptions
    map_provider_factory: MapProviderFactory | None = None
    feature_support: DatasetFeatureSupport = DatasetFeatureSupport()
    split_support: DatasetSplitSupport = DatasetSplitSupport()
    temporal_support: DatasetTemporalSupport | None = None
    tasks: Mapping[str, PredictionTaskConfig] = field(
        default_factory=dict[str, PredictionTaskConfig],
    )
    """Named prediction tasks supported by this dataset."""
    default_task: str | None = None
    """Named prediction task selected by default, if any."""

    def __post_init__(self) -> None:
        """Validate descriptor-owned tasks against the default scene horizon."""
        if self.default_task is not None and self.default_task not in self.tasks:
            msg = f"Default task '{self.default_task}' is not defined for dataset '{self.name}'."
            raise DatasetRegistryError(msg)
        for name, task in self.tasks.items():
            if task.prediction_end > self.default_config.scenes.horizon_frames:
                msg = (
                    f"Task '{name}' for dataset '{self.name}' ends at "
                    f"{task.prediction_end}, beyond the default horizon of "
                    f"{self.default_config.scenes.horizon_frames}."
                )
                raise DatasetRegistryError(msg)

    def parse_loader_options(self, payload: Mapping[str, object] | None) -> LoaderOptionsModel:
        """Parse and validate dataset-owned config from plain data."""
        try:
            return self.loader_options_model.parse(dict(payload or {}))
        except ValidationError as exc:
            msg = f"Invalid loader options for dataset '{self.name}': {exc}"
            raise LoaderConfigError(msg) from exc

    @contextmanager
    def open_resources(self, root: Path, request: LoaderPlan) -> Generator[MapProvider | None]:
        """Open the run-scoped map provider for this dataset, if any."""
        if self.map_provider_factory is None:
            yield None
            return
        with self.map_provider_factory(root, request.scenes, request.map) as map_provider:
            yield map_provider

    def build_loader(
        self,
        *,
        root: Path,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> SceneLoader[Any, Any]:
        """Construct one loader instance for this dataset descriptor."""
        return self.loader_cls.from_loader_request(
            data_root=root,
            request=request,
            map_provider=map_provider,
        )


@dataclass(frozen=True, slots=True)
class _BuiltinDatasetDescriptor:
    module: str
    export_name: str = "DATASET_DESCRIPTOR"
    export_key: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    extra: str | None = None


def _builtin(
    module: str,
    *,
    export_name: str = "DATASET_DESCRIPTOR",
    export_key: str | None = None,
    optional_dependencies: tuple[str, ...] = (),
    extra: str | None = None,
) -> _BuiltinDatasetDescriptor:
    return _BuiltinDatasetDescriptor(
        module=module,
        export_name=export_name,
        export_key=export_key,
        optional_dependencies=optional_dependencies,
        extra=extra,
    )


_BUILTIN_DATASETS: dict[str, _BuiltinDatasetDescriptor] = {
    "a43": _builtin("prejectory.datasets.a43"),
    "ad4che": _builtin(
        "prejectory.datasets.ad4che",
        optional_dependencies=("cv2",),
        extra="ad4che",
    ),
    "apolloscape": _builtin("prejectory.datasets.apolloscape"),
    "argoverse1": _builtin("prejectory.datasets.argoverse1"),
    "argoverse2": _builtin("prejectory.datasets.argoverse2"),
    "eth": _builtin(
        "prejectory.datasets.eth_ucy",
        export_name="DATASET_DESCRIPTORS",
        export_key="eth",
    ),
    "hotel": _builtin(
        "prejectory.datasets.eth_ucy",
        export_name="DATASET_DESCRIPTORS",
        export_key="hotel",
    ),
    "univ": _builtin(
        "prejectory.datasets.eth_ucy",
        export_name="DATASET_DESCRIPTORS",
        export_key="univ",
    ),
    "zara1": _builtin(
        "prejectory.datasets.eth_ucy",
        export_name="DATASET_DESCRIPTORS",
        export_key="zara1",
    ),
    "zara2": _builtin(
        "prejectory.datasets.eth_ucy",
        export_name="DATASET_DESCRIPTORS",
        export_key="zara2",
    ),
    "exid": _builtin(
        "prejectory.datasets.levelx",
        export_name="DATASET_DESCRIPTORS",
        export_key="exid",
    ),
    "highd": _builtin(
        "prejectory.datasets.levelx",
        export_name="DATASET_DESCRIPTORS",
        export_key="highd",
    ),
    "i80": _builtin(
        "prejectory.datasets.ngsim",
        export_name="DATASET_DESCRIPTORS",
        export_key="i80",
    ),
    "ind": _builtin(
        "prejectory.datasets.levelx",
        export_name="DATASET_DESCRIPTORS",
        export_key="ind",
    ),
    "interaction": _builtin("prejectory.datasets.interaction"),
    "nuscenes": _builtin("prejectory.datasets.nuscenes"),
    "opendd": _builtin("prejectory.datasets.opendd"),
    "round": _builtin(
        "prejectory.datasets.levelx",
        export_name="DATASET_DESCRIPTORS",
        export_key="round",
    ),
    "sind": _builtin("prejectory.datasets.sind"),
    "unid": _builtin(
        "prejectory.datasets.levelx",
        export_name="DATASET_DESCRIPTORS",
        export_key="unid",
    ),
    "us101": _builtin(
        "prejectory.datasets.ngsim",
        export_name="DATASET_DESCRIPTORS",
        export_key="us101",
    ),
    "vod": _builtin("prejectory.datasets.vod"),
    "waymo": _builtin(
        "prejectory.datasets.waymo",
        optional_dependencies=("google.protobuf",),
        extra="waymo",
    ),
}

_BUILTIN_DATASET_IDS: dict[str, int] = {
    name: dataset_id for dataset_id, name in enumerate(_BUILTIN_DATASETS.keys())
}


def dataset_id_for_name(name: str) -> int | None:
    """Return the stable built-in dataset id for a dataset name, if known."""
    return _BUILTIN_DATASET_IDS.get(name)


def dataset_names_by_id() -> tuple[str, ...]:
    """Return built-in dataset names ordered by their stable dataset ids."""
    return tuple(_BUILTIN_DATASET_IDS.keys())


def register_dataset(descriptor: DatasetDescriptor) -> None:
    """Register one dataset descriptor in the in-memory registry.

    This is the main extension point for adding new datasets to prejectory from
    an external module.

    Parameters
    ----------
    descriptor : DatasetDescriptor
        The dataset descriptor to register.
    """
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise DatasetRegistryError(msg)
    _REGISTRY[descriptor.name] = descriptor
    logger.debug("Registered dataset descriptor", extra={"dataset": descriptor.name})


def get_dataset(name: str) -> DatasetDescriptor:
    """Return one registered or built-in dataset descriptor.

    Parameters
    ----------
    name : str
        The name of the dataset to resolve. The name should match the `name`
        field of the returned descriptor, and is case-sensitive.

    Returns
    -------
    DatasetDescriptor
        The resolved dataset descriptor.

    """
    if name in _REGISTRY:
        logger.debug("Resolved dataset descriptor from in-memory registry", extra={"dataset": name})
        return _REGISTRY[name]

    if name not in _BUILTIN_DATASETS:
        raise DatasetNotFoundError(name, list_datasets())

    builtin = _BUILTIN_DATASETS[name]
    missing = _missing_optional_dependencies(builtin)
    if missing:
        raise _missing_dependency_error(
            subject=f"Dataset '{name}'",
            builtin=builtin,
            missing=missing,
        )

    logger.debug("Resolved dataset descriptor from built-in registry", extra={"dataset": name})
    return _load_builtin_descriptor(name)


def list_datasets() -> list[str]:
    """Return the sorted list of available dataset names.

    Returns
    -------
    list[str]
        The sorted list of available dataset names, including both registered
        and built-in datasets that have their optional dependencies satisfied.
    """
    builtin_names = {
        name
        for name, builtin in _BUILTIN_DATASETS.items()
        if not _missing_optional_dependencies(builtin)
    }
    return sorted(set(_REGISTRY) | builtin_names)


@functools.cache
def _load_builtin_descriptor(name: str) -> DatasetDescriptor:
    builtin = _BUILTIN_DATASETS.get(name)
    if builtin is None:
        raise DatasetNotFoundError(name, list_datasets())

    module = importlib.import_module(builtin.module)
    try:
        exported = getattr(module, builtin.export_name)
    except AttributeError as exc:
        msg = (
            f"Built-in dataset module '{builtin.module}' does not export '{builtin.export_name}' "
            f"for dataset '{name}'."
        )
        raise DatasetRegistryError(msg) from exc

    descriptor = exported if builtin.export_key is None else exported[builtin.export_key]
    if not isinstance(descriptor, DatasetDescriptor):
        msg = f"Built-in dataset '{name}' did not resolve to a DatasetDescriptor."
        raise DatasetRegistryError(msg)
    if descriptor.name != name:
        msg = (
            f"Built-in dataset '{name}' resolved to descriptor '{descriptor.name}'. "
            "Descriptor names must match builtin registry keys."
        )
        raise DatasetRegistryError(msg)
    return descriptor


@functools.lru_cache
def _missing_optional_dependencies(builtin: _BuiltinDatasetDescriptor) -> tuple[str, ...]:
    return tuple(
        module_name for module_name in builtin.optional_dependencies if not _has_module(module_name)
    )


@functools.lru_cache
def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_dependency_error(
    *,
    subject: str,
    builtin: _BuiltinDatasetDescriptor,
    missing: tuple[str, ...],
) -> MissingOptionalDependencyError:
    install_target = f"prejectory[{builtin.extra}]" if builtin.extra else None
    install_hint = f"Install {install_target} to use it." if install_target else ""
    missing_str = ", ".join(missing)
    msg = (
        f"{subject} is unavailable because the following optional dependencies are missing: "
        f"{missing_str}. {install_hint}"
    )
    return MissingOptionalDependencyError(
        msg,
        dependencies=tuple(missing),
        install_target=install_target,
    )
