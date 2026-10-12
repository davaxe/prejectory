"""Public runtime data models shared by API and internal execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # ruff: ignore[typing-only-standard-library-import] - Pydantic resolves this forward reference at runtime.
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from prejectory.config.models import (
    DatasetConfigPatch,
    PredictionTaskConfig,
    ScreeningConfig,
    effective_prediction_bounds,
    effective_scene_window,
)
from prejectory.config.parse import ProjectConfig  # ruff: ignore[typing-only-first-party-import] - Pydantic resolves the request model at runtime.
from prejectory.core.errors import ConfigurationError
from prejectory.core.scene.model import derived_trajectory_fields
from prejectory.core.scene.schema import POSITIONS_ONLY, TrajectorySchema, get_trajectory_schema
from prejectory.datasets.registry import dataset_id_for_name, dataset_names_by_id
from prejectory.io.base import StorageBackend
from prejectory.io.manifest import DatasetManifest, PredictionTaskManifest, package_version
from prejectory.processing.models import LoaderPlan, ReadSelection, SplitAssignmentPlan
from prejectory.processing.screening.agent import AgentRequireFrames
from prejectory.processing.screening.base import PassingRequirement

if TYPE_CHECKING:
    from prejectory.config.models import DatasetConfig, OutputConfig
    from prejectory.datasets.registry import DatasetDescriptor
    from prejectory.io.records import PredictionBounds
    from prejectory.processing.loading.models import LoaderOptionsModel
    from prejectory.runtime.state import ExecutionStats


@dataclass(frozen=True, slots=True)
class CleanupRemovalSummary:
    """Aggregate cleanup-removal statistics over candidate scenes with cleanup stats.

    Cleanup is recorded before screening rejection is applied, so `scene_count`
    does not necessarily match the number of written scenes.
    """

    scene_count: int
    total_rows_removed: int
    average_rows_removed_per_scene: float
    min_rows_removed_per_scene: int
    max_rows_removed_per_scene: int
    total_agents_removed: int
    average_agents_removed_per_scene: float
    min_agents_removed_per_scene: int
    max_agents_removed_per_scene: int


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    """Final cleanup statistics, optionally broken down by cleanup rule."""

    overall: CleanupRemovalSummary
    by_rule: dict[str, CleanupRemovalSummary]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final result of one dataset execution.

    `stats` is the canonical source, scene, split, screening, and cleanup
    progress-counter payload. `cleanup_summary` contains the heavier final
    diagnostic cleanup summary.
    """

    dataset: str
    """Dataset that was executed."""
    output_dir: Path
    """Directory where output was written."""
    storage_backend: StorageBackend
    """Storage backend used for writing output."""
    stats: ExecutionStats
    """Final source, scene, split, screening, and cleanup counters."""
    scene_limit: int | None
    """Output scene limit used for this run, if any."""
    cleanup_summary: CleanupSummary | None
    """Final cleanup summary over candidate scenes with cleanup statistics."""
    elapsed_time_seconds: float
    """Wall-clock execution time in seconds."""

    @property
    def written_scenes(self) -> int:
        """Number of scenes successfully written."""
        return self.stats.written_scenes


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Fully resolved runtime plan produced from an execution request.

    !!! info "Do not construct this directly"
        This is not meant to be directly constructed. Instead, it should be
        built using the
        [`resolve_request`][prejectory.runtime.api.resolve_request] function,
        which takes [`ExecutionRequest`][prejectory.runtime.ExecutionRequest] and
        produces a fully resolved plan ready for execution.

    """

    _descriptor: DatasetDescriptor = field(repr=False)
    """Resolved dataset descriptor."""
    selected_task: str | None
    """Named descriptor prediction task selected for this run, if any."""
    input_dir: Path
    """Input dataset root."""
    output_dir: Path
    """Output dataset root."""
    storage_backend: StorageBackend
    """Storage backend selected for writing output records."""
    config: DatasetConfig
    """Dataset config after defaults, config files, and overrides are merged."""
    _loader: LoaderPlan = field(repr=False)
    """Loader-facing subset of the resolved configuration."""
    _assignment: SplitAssignmentPlan = field(repr=False)
    """Compiled split-assignment request."""
    effective_horizon_frames: int
    """Horizon frame count after resampling/window configuration is applied."""
    effective_prediction_bounds: PredictionBounds | None
    """Half-open prediction bounds after resampling, if configured."""
    effective_sample_time: float
    """Effective `sample_time` interval in seconds after resampling is applied."""
    limit: int | None = None
    """Optional maximum number of scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""
    overwrite: bool = False
    """Whether execution may replace an existing non-empty output directory."""

    diagnostics: tuple[str, ...] = ()
    """Planning warnings that do not prevent execution."""

    @property
    def include_map(self) -> bool:
        """Whether this run includes map data."""
        return self._loader.map is not None

    def summary(self) -> str:
        """Return a concise, framework-neutral description of the resolved run."""
        task = self.selected_task or ("custom" if self.config.task else "none")
        return (
            f"{self.dataset}: {self.input_dir} -> {self.output_dir}\n"
            f"{self.storage_backend.value}; task={task}; "
            f"{self.effective_horizon_frames} frames at {self.effective_sample_time:g}s; "
            f"{self.config.runtime.jobs} worker(s)"
        )

    def __post_init__(self) -> None:
        """Validate the runtime plan after initialization."""
        if self.limit == 0:
            object.__setattr__(self, "limit", None)
        if self.limit is not None and self.limit <= 0:
            msg = f"Limit must be a positive integer, got {self.limit}."
            raise ConfigurationError(msg)

    @property
    def dataset(self) -> str:
        """Dataset key for this plan."""
        return self._descriptor.name

    @property
    def parallel(self) -> bool:
        """Whether the runtime plan requests parallel execution."""
        return self.config.runtime.jobs > 1

    @property
    def output_config(self) -> OutputConfig:
        """Resolved output configuration."""
        return self.config.output

    @property
    def trajectory_schema(self) -> TrajectorySchema:
        """Resolved output trajectory schema."""
        return get_trajectory_schema(self.output_config.trajectory_schema)

    def manifest(self) -> DatasetManifest:
        """Return the dataset manifest for this plan."""
        export_config = self.output_config
        derivation_source = trajectory_schema_after_transforms(
            self._descriptor.native_schema,
            self.config,
        )
        source_task = self.config.task
        effective_bounds = self.effective_prediction_bounds
        prediction_task = (
            None
            if source_task is None or effective_bounds is None
            else PredictionTaskManifest(
                name=self.selected_task,
                source_prediction_origin=source_task.prediction_origin,
                source_prediction_end=source_task.prediction_end,
                prediction_origin=effective_bounds.prediction_origin,
                prediction_end=effective_bounds.prediction_end,
            )
        )
        return DatasetManifest(
            dataset=self.dataset,
            dataset_names=(
                dataset_names_by_id()
                if dataset_id_for_name(self.dataset) is not None
                else (self.dataset,)
            ),
            storage_backend=self.storage_backend.value,
            prejectory_version=package_version(),
            precision=export_config.precision,
            feature_columns=self.trajectory_schema.feature_columns(),
            trajectory_schema=self.trajectory_schema.name,
            trajectory_schema_fields=self.trajectory_schema.semantic_fields(),
            recenter_positions=export_config.recenter_positions,
            source_trajectory_schema=self._descriptor.native_schema.name,
            source_trajectory_schema_fields=self._descriptor.native_schema.semantic_fields(),
            sample_time=self.effective_sample_time,
            original_sample_time=self.config.scenes.sample_time,
            horizon_frames=self.effective_horizon_frames,
            prediction_task=prediction_task,
            has_map=self.include_map,
            derived_features=tuple(
                field.to_str()
                for field in derived_trajectory_fields(
                    derivation_source,
                    self.trajectory_schema,
                    sample_time=self.config.scenes.sample_time,
                )
            ),
        )


def build_loader_plan(
    *,
    descriptor: DatasetDescriptor,
    resolved_config: DatasetConfig,
    include_map: bool | None,
) -> LoaderPlan:
    """Compile the loader-facing request for one resolved dataset config."""
    loader_options: LoaderOptionsModel = descriptor.parse_loader_options(
        resolved_config.loader_options,
    )
    map_config = (
        None
        if (include_map is False or not descriptor.feature_support.map)
        else resolved_config.map
    )
    screening = _screening_with_task_endpoint(resolved_config.screening, resolved_config.task)
    return LoaderPlan(
        scenes=resolved_config.scenes,
        screening=screening,
        read=ReadSelection.from_config(
            resolved_config.read,
            supported_native_splits=descriptor.supported_native_splits,
        ),
        loader_options=loader_options,
        map=map_config,
    )


def _screening_with_task_endpoint(
    screening: ScreeningConfig | None,
    task: PredictionTaskConfig | None,
) -> ScreeningConfig | None:
    if task is None or not task.require_history_endpoint:
        return screening

    base = screening or ScreeningConfig()
    agents = dict(base.agents)
    agents["prediction_history_endpoint"] = AgentRequireFrames.define(
        [task.prediction_origin - 1],
        require=PassingRequirement(absolute=1),
    )
    return ScreeningConfig(cleanup=base.cleanup, scenes=base.scenes, agents=agents)


class ExecutionRequest(BaseModel):
    """User-facing request for one dataset processing run.

    The request is intentionally small: it names the dataset, input/output
    paths, optional config file, and runtime overrides. Pass it to
    [`resolve_request`][prejectory.runtime.api.resolve_request] to obtain an
    [`ExecutionPlan`][prejectory.runtime.ExecutionPlan], or to
    [`execute_request`][prejectory.runtime.api.execute_request] to run directly.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    dataset: str
    """Dataset registry key, such as `a43` or `waymo`."""
    input_dir: Path
    """Root directory containing the raw dataset files."""
    output_dir: Path
    """Directory where processed output should be written."""
    storage_backend: StorageBackend | str = StorageBackend.PICKLE
    """Output storage backend. Built-in values are `pickle` and `mds`."""
    config: ProjectConfig | Path | str | None = None
    """Project configuration or TOML path, layered on dataset defaults."""
    overrides: DatasetConfigPatch = Field(default_factory=DatasetConfigPatch)
    """Programmatic runtime overrides applied after the config file."""
    task: str | PredictionTaskConfig | None = None
    """Omit to inherit, use a name or inline task to replace, or None to disable."""
    include_map: bool | None = None
    """Override for map output. `None` uses the resolved dataset config."""
    limit: int | None = None
    """Optional maximum number of scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""
    overwrite: bool = False
    """Whether execution may replace an existing non-empty output directory."""
    input_dir_exists: bool = True
    """Whether request resolution should require `input_dir` to exist."""


def resolve_effective_scene_window(
    config: DatasetConfig,
) -> tuple[int, tuple[int, int] | None, float]:
    """Return the effective horizon, prediction bounds, and sample time."""
    horizon, sample_time = effective_scene_window(config.scenes)
    return horizon, effective_prediction_bounds(config), sample_time


def trajectory_schema_after_transforms(
    native_schema: TrajectorySchema,
    config: DatasetConfig,
) -> TrajectorySchema:
    """Return fields that remain semantically valid after temporal transforms."""
    resample = config.scenes.resample
    if resample is None:
        return native_schema
    if (
        resample.up == 1
        and resample.down == 1
        and not resample.emit_velocity
        and not resample.emit_acceleration
    ):
        return native_schema
    return POSITIONS_ONLY
