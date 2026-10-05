"""Configuration models for dataset generation."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

import numpy as np
from pydantic import BeforeValidator, Field, field_validator, model_validator
from typing_extensions import override

from prejectory.config.base import (
    Clear,
    Clearable,
    ConfigBase,
    ConfigPatch,
    DictPatch,
    MappingPatch,
    ResampleMethod,
    ResolvedConfig,
    apply_optional,
)
from prejectory.core.categories import DatasetSplit, EdgeType, EdgeTypeLike, coerce_edge_types
from prejectory.core.errors import ConfigurationError
from prejectory.core.functional.window import WindowPolicy  # ruff: ignore[typing-only-first-party-import]
from prejectory.core.scene import CANONICAL, TrajectorySchema
from prejectory.core.scene.schema import TrajectorySchemaDefinition
from prejectory.processing.screening.agent import AgentCheckRule  # ruff: ignore[typing-only-first-party-import]
from prejectory.processing.screening.cleanup import CleanupRule  # ruff: ignore[typing-only-first-party-import]
from prejectory.processing.screening.scene import SceneCheckRule  # ruff: ignore[typing-only-first-party-import]

if TYPE_CHECKING:
    from prejectory.core.typing import T


EdgeTypes = Annotated[
    frozenset[EdgeType],
    BeforeValidator(lambda v: coerce_edge_types(v, frozenset)),
]

FloatDType = type[np.float32] | type[np.float64]
OutputPrecision = Literal["float32", "float64"]
"""Accepted floating-point precision labels for serialized numeric arrays.

The chosen label is later resolved to the corresponding NumPy dtype when
writers materialize trajectory tensors.
"""

TrajectorySchemaLike = TrajectorySchema | str | TrajectorySchemaDefinition
"""User-facing trajectory schema inputs accepted by output-related config.

Callers may provide a registered schema object directly, reference a schema by
name, or inline a full schema definition that resolves to a
[`TrajectorySchema`][prejectory.core.scene.TrajectorySchema].
"""

ReadStrategy = Literal["all", "native"]
"""Supported runtime read-strategy override names accepted by the CLI layer.

The string values map directly to the concrete read config models in
[`prejectory.config.models`][].
"""

AssignStrategy = Literal["none", "preserve-native", "scene", "source", "time", "shuffled-time"]
"""Supported runtime assignment-strategy override names accepted by the CLI layer.

The string values map directly to the concrete assignment config models in
[`prejectory.config.models`][].
"""


class SplitWeights(ResolvedConfig):
    """Weights used when routing data into train/val/test assignments."""

    train: float = Field(ge=0, default=0.0)
    """Weight assigned to the training split."""
    val: float = Field(ge=0, default=0.0)
    """Weight assigned to the validation split."""
    test: float = Field(ge=0, default=0.0)
    """Weight assigned to the test split."""

    @model_validator(mode="after")
    def _validate_sum(self) -> SplitWeights:
        """Validate that at least one split weight is positive."""
        total = self.train + self.val + self.test
        if total == 0:
            v = f"train={self.train}, val={self.val}, test={self.test}"
            msg = f"At least one split weight must be greater than 0, got {v}"
            raise ConfigurationError(msg)
        return self


class ReadAll(ResolvedConfig):
    """Read the full dataset input surface."""

    strategy: Literal["all"] = Field("all", repr=False, init=False)


class ReadNative(ResolvedConfig):
    """Read only selected dataset-native partitions."""

    strategy: Literal["native"] = Field("native", repr=False, init=False)
    splits: frozenset[DatasetSplit] | None = None
    """Native dataset partitions to read from the DatasetSource dataset."""


ReadConfig = Annotated[ReadAll | ReadNative, Field(discriminator="strategy")]


class NoAssign(ResolvedConfig):
    """No output split assignment."""

    strategy: Literal["none"] = Field("none", repr=False, init=False)


class PreserveNativeAssign(ResolvedConfig):
    """Preserve dataset-native split labels in output."""

    strategy: Literal["preserve-native"] = Field("preserve-native", repr=False, init=False)


class SceneAssign(ResolvedConfig):
    """Scene-based output split assignment."""

    strategy: Literal["scene"] = Field("scene", repr=False, init=False)
    ratio: SplitWeights


class SourceAssign(ResolvedConfig):
    """DatasetSource-based output split assignment."""

    strategy: Literal["source"] = Field("source", repr=False, init=False)
    ratio: SplitWeights


class TimeBlockAssign(ResolvedConfig):
    """Time-block output split assignment."""

    gap: int = Field(ge=0, default=0)
    strategy: Literal["time"] = Field("time", repr=False, init=False)
    ratio: SplitWeights


class ShuffledTimeBlockAssign(ResolvedConfig):
    """Shuffled time-block output split assignment."""

    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    strategy: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)
    ratio: SplitWeights


AssignConfig = Annotated[
    NoAssign
    | PreserveNativeAssign
    | SceneAssign
    | SourceAssign
    | TimeBlockAssign
    | ShuffledTimeBlockAssign,
    Field(discriminator="strategy"),
]


class RuntimeConfig(ResolvedConfig):
    """Execution controls for processing requests.

    Attributes
    ----------
    jobs : int | Literal["auto"]
        Number of worker processes to use. Set to `"auto"` to let runtime
        choose an appropriate value.
    chunksize : int | None
        Optional per-worker batch size for scene dispatch.
    """

    jobs: int = Field(default=1, gt=0)
    """Number of worker processes to use."""
    chunksize: int | None = Field(default=None, gt=0)
    """Optional per-worker batch size for scene dispatch."""


class RuntimePatch(ConfigPatch[RuntimeConfig]):
    """Patch model for overriding :class:`RuntimeConfig`."""

    jobs: int | Literal["auto"] | None = None
    """Replacement worker count or `"auto"` to use the current CPU count."""
    chunksize: int | None = Field(default=None, gt=0)
    """Replacement per-worker batch size for scene dispatch."""
    full_config_type: type[RuntimeConfig] = Field(default=RuntimeConfig, init=False, repr=False)

    @override
    def merge_into(self, target: RuntimeConfig | None) -> RuntimeConfig:
        partial = self.model_copy(update={"jobs": mp.cpu_count()}) if self.jobs == "auto" else self
        return ConfigPatch[RuntimeConfig].merge_into(partial, target)


class MDSOutputConfig(ResolvedConfig):
    """Backend-specific tuning for Mosaic Streaming output."""

    compression: str | None = None
    """Compression algorithm to use for each Mosaic shard, if any."""
    hashes: tuple[str, ...] | None = None
    """Hash algorithms recorded for each shard, if hashing is enabled."""
    size_limit: str | int = 67_108_864
    """Maximum shard size accepted by the writer before starting a new shard."""
    exist_ok: bool = False
    """Whether an existing output location may be reused instead of raising an error."""


class MDSOutputPatch(ConfigPatch[MDSOutputConfig]):
    """Patch model for overriding Mosaic Streaming writer settings."""

    compression: str | None = None
    """Replacement compression algorithm for Mosaic shards."""
    hashes: tuple[str, ...] | None = None
    """Replacement hash algorithms recorded for Mosaic shards."""
    size_limit: str | int | None = None
    """Replacement maximum shard size for the Mosaic writer."""
    exist_ok: bool | None = None
    """Replacement policy for whether existing output locations are allowed."""
    full_config_type: type[MDSOutputConfig] = Field(default=MDSOutputConfig, init=False, repr=False)


class ZarrOutputConfig(ResolvedConfig):
    """Backend-specific chunking and compression for Zarr output."""

    scene_chunk: int = Field(default=256, gt=0)
    agent_chunk: int = Field(default=4096, gt=0)
    map_node_chunk: int = Field(default=16384, gt=0)
    map_edge_chunk: int = Field(default=16384, gt=0)
    compression_level: int | None = Field(default=3, ge=0, le=22)


class ZarrOutputPatch(ConfigPatch[ZarrOutputConfig]):
    """Patch model for Zarr chunking and compression settings."""

    scene_chunk: int | None = Field(default=None, gt=0)
    agent_chunk: int | None = Field(default=None, gt=0)
    map_node_chunk: int | None = Field(default=None, gt=0)
    map_edge_chunk: int | None = Field(default=None, gt=0)
    compression_level: int | None = Field(default=None, ge=0, le=22)
    full_config_type: type[ZarrOutputConfig] = Field(
        default=ZarrOutputConfig, init=False, repr=False
    )


class OutputConfig(ResolvedConfig):
    """Resolved output configuration shared by storage backends."""

    trajectory_schema: TrajectorySchemaLike = Field(default=CANONICAL)
    """Trajectory schema used when encoding scene records."""
    precision: OutputPrecision = "float32"
    """Floating-point precision used for serialized numeric arrays."""
    recenter_positions: bool = True
    """Whether scene positions are translated into a local origin before writing."""
    mds: MDSOutputConfig = Field(default_factory=MDSOutputConfig)
    """Backend-specific tuning for Mosaic Streaming outputs."""
    zarr: ZarrOutputConfig = Field(default_factory=ZarrOutputConfig)
    """Backend-specific tuning for Zarr outputs."""


class OutputPatch(ConfigPatch[OutputConfig]):
    """Patch model for overriding shared output settings."""

    trajectory_schema: TrajectorySchemaLike | None = Field(default=None)
    """Replacement trajectory schema used when encoding scene records."""
    precision: OutputPrecision | None = None
    """Replacement floating-point precision for serialized numeric arrays."""
    recenter_positions: bool | None = None
    """Replacement policy for recentering scene positions before writing."""
    mds: MDSOutputPatch | None = None
    """Backend-specific patch overrides for Mosaic Streaming outputs."""
    zarr: ZarrOutputPatch | None = None
    """Backend-specific patch overrides for Zarr outputs."""
    full_config_type: type[OutputConfig] = Field(default=OutputConfig, init=False, repr=False)


class SceneExtentExtraction(ResolvedConfig):
    """Configuration for extraction around the scene trajectory extent."""

    mode: Literal["scene_extent"] = Field("scene_extent", repr=False, init=False)
    padding: float = Field(ge=1.0, default=1.0)
    """Scale factor applied around the scene extent before cropping the map."""
    shape: Literal["circle", "bounding_box"] = Field(default="circle")
    """Adaptive crop shape used around the scene extent."""


class CircularExtraction(ResolvedConfig):
    """Configuration for circular map extraction mode."""

    mode: Literal["circle"] = Field("circle", repr=False, init=False)
    radius: float = Field(gt=0)
    """Radius of the circular crop centered on the scene in map units."""


class BoundingBoxExtraction(ResolvedConfig):
    """Configuration for bounding-box map extraction mode."""

    mode: Literal["bounding_box"] = Field("bounding_box", repr=False, init=False)
    width: float = Field(gt=0)
    """Width of the extracted map crop in map units."""
    height: float = Field(gt=0)
    """Height of the extracted map crop in map units."""


class TrajectoryBufferExtraction(ResolvedConfig):
    """Configuration for buffering the scene trajectories directly."""

    mode: Literal["trajectory_buffer"] = Field("trajectory_buffer", repr=False, init=False)
    radius: float = Field(gt=0)
    """Buffer radius around each relevant trajectory point in map units."""


class FullMapExtraction(ResolvedConfig):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


class MapEdgeTypeRules(ResolvedConfig):
    """Semantic edge-type filtering and remapping rules."""

    include: EdgeTypes | None = Field(default=None)
    """Optional allow-list of edge types to keep after remapping."""
    exclude: EdgeTypes = frozenset()
    """Edge types to drop after remapping."""
    remap: dict[EdgeType, EdgeType] = Field(default_factory=dict)
    """Mapping applied to edge types before include/exclude filters."""

    @model_validator(mode="after")
    def _validate_no_conflicts(self) -> MapEdgeTypeRules:
        if self.include is not None and self.exclude.intersection(self.include):
            overlap = self.exclude.intersection(self.include)
            msg = f"Conflict in edge type rules: {overlap}."
            raise ValueError(msg)
        if self.remap.keys() & self.remap.values():
            overlap = self.remap.keys() & self.remap.values()
            msg = f"Conflict in edge type remapping: {overlap}."
            raise ValueError(msg)
        return self

    @field_validator("remap", mode="before")
    @classmethod
    def _coerce_remap_keys_and_values(
        cls,
        v: dict[EdgeTypeLike, EdgeTypeLike],
    ) -> dict[EdgeType, EdgeType]:
        return {EdgeType.from_value(k): EdgeType.from_value(val) for k, val in v.items()}


MapExtraction = Annotated[
    CircularExtraction
    | BoundingBoxExtraction
    | FullMapExtraction
    | SceneExtentExtraction
    | TrajectoryBufferExtraction,
    Field(discriminator="mode"),
]
"""Discriminated union of the supported map extraction strategies.

The `mode` field selects whether map geometry is cropped around the scene,
within a circle, within a bounding box, around the trajectory points directly,
or retained in full.
"""


class MapConfig(ResolvedConfig):
    """Configuration for map data processing."""

    min_distance: float | None = Field(gt=0, default=2)
    """Minimum spacing allowed between neighboring map points after simplification."""
    interpolation_distance: float | None = Field(gt=0, default=5.0)
    """Target spacing used when interpolating map geometry."""
    extraction: MapExtraction = Field(default_factory=FullMapExtraction)
    """Map extraction strategy used to crop or retain DatasetSource map geometry."""
    edge_types: MapEdgeTypeRules | None = Field(default=None)
    """Optional edge-type remapping and filtering rules."""

    @model_validator(mode="after")
    def _validate_distances(self) -> MapConfig:
        if self.interpolation_distance is None or self.min_distance is None:
            return self
        if self.interpolation_distance < self.min_distance:
            msg = (
                f"interpolation_distance ({self.interpolation_distance}) must be greater "
                f"than or equal to min_distance ({self.min_distance})."
            )
            raise ValueError(msg)
        return self


class MapEdgeTypeRulesPatch(ConfigPatch[MapEdgeTypeRules]):
    """Patch model for overriding :class:`MapEdgeTypeRules`."""

    include: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement allow-list of edge types to keep."""
    exclude: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement deny-list of edge types to drop."""
    remap: dict[EdgeTypeLike, EdgeTypeLike] | None = Field(default=None)
    """Replacement edge-type remapping rules."""
    full_config_type: type[MapEdgeTypeRules] = Field(MapEdgeTypeRules, repr=False, init=False)


class MapPatch(ConfigPatch[MapConfig]):
    """Patch model for overriding :class:`MapConfig`."""

    min_distance: float | None = Field(gt=0, default=None)
    """Replacement minimum spacing for simplified map points."""
    interpolation_distance: float | None = Field(gt=0, default=None)
    """Replacement interpolation spacing for map geometry."""
    extraction: MapExtraction | None = Field(default=None)
    """Replacement map extraction strategy."""
    edge_types: Clearable[MapEdgeTypeRulesPatch] = Field(default=None)
    """Replacement edge-type rules, or `false` to clear inherited rules."""
    full_config_type: type[MapConfig] = Field(MapConfig, repr=False, init=False)

    @override
    def merge_into(self, target: MapConfig | None) -> MapConfig:
        base = MapConfig() if target is None else target
        return MapConfig(
            min_distance=(
                self.min_distance if "min_distance" in self.model_fields_set else base.min_distance
            ),
            interpolation_distance=(
                self.interpolation_distance
                if "interpolation_distance" in self.model_fields_set
                else base.interpolation_distance
            ),
            extraction=self.extraction if self.extraction is not None else base.extraction,
            edge_types=apply_optional(self.edge_types, base.edge_types),
        )


class ResampleConfig(ResolvedConfig):
    """Validated specification for temporal resampling."""

    up: int = Field(default=1, gt=0)
    """Upsampling factor applied before downsampling."""
    down: int = Field(default=1, gt=0)
    """Downsampling factor applied after upsampling."""
    method: ResampleMethod = "linear"
    """Interpolation method used during resampling."""
    coordinates: tuple[str, ...] = Field(default=("x", "y"))
    """Coordinate fields resampled by the interpolation step."""
    emit_velocity: bool = Field(default=False)
    """Whether velocity derivatives should be emitted during resampling."""
    emit_acceleration: bool = Field(default=False)
    """Whether acceleration derivatives should be emitted during resampling."""
    max_gap: int = Field(default=1, gt=0)
    """Maximum consecutive missing frames allowed when interpolating observations."""

    @model_validator(mode="after")
    def _validate(self) -> ResampleConfig:
        if self.method == "linear" and (self.emit_velocity or self.emit_acceleration):
            msg = "Linear resampling does not support emitting derivatives."
            raise ConfigurationError(msg)
        return self


class ResamplePatch(ConfigPatch[ResampleConfig]):
    """Patch model for overriding temporal resampling settings."""

    up: int | None = None
    """Replacement upsampling factor."""
    down: int | None = None
    """Replacement downsampling factor."""
    method: ResampleMethod | None = None
    """Replacement interpolation method."""
    coordinates: tuple[str, ...] | None = None
    """Replacement coordinate fields to resample."""
    emit_velocity: bool | None = None
    """Replacement policy for emitting velocity derivatives."""
    emit_acceleration: bool | None = None
    """Replacement policy for emitting acceleration derivatives."""
    max_gap: int | None = None
    """Replacement maximum consecutive gap allowed during interpolation."""
    full_config_type: type[ResampleConfig] = Field(default=ResampleConfig, init=False, repr=False)


class WindowConfig(ResolvedConfig):
    """Configuration for sliding-window extraction of scenes."""

    step: int = Field(gt=0)
    """Stride between consecutive scene windows in frames."""
    policy: WindowPolicy = "strict"
    """Completeness policy for sources that do not fully cover a window."""


class WindowPatch(ConfigPatch[WindowConfig]):
    """Patch model for overriding sliding-window extraction settings."""

    step: int | None = None
    """Replacement stride between consecutive sampled windows in frames."""
    policy: WindowPolicy | None = None
    """Replacement completeness policy for incomplete windows."""
    full_config_type: type[WindowConfig] = Field(default=WindowConfig, init=False, repr=False)


class LaneChangeConfig(ResolvedConfig):
    """Configuration for lane-change-aware sampling."""

    persist: int = Field(gt=0)
    """Minimum number of frames a lane-change state must persist to count."""
    margin_before: int = Field(default=0, ge=0)
    """Extra context frames to keep before the detected lane change."""
    margin_after: int = Field(default=0, ge=0)
    """Extra context frames to keep after the detected lane change."""
    required_lane_changes: int = Field(default=1, gt=0)
    """Minimum number of lane changes required for a positive scene window."""
    negative_keep_every: int = Field(default=3, ge=1)
    """Keep one negative scene window out of every N candidates."""


class LaneChangePatch(ConfigPatch[LaneChangeConfig]):
    """Patch model for overriding lane-change-aware sampling settings."""

    persist: int | None = None
    """Replacement persistence threshold for lane-change detection."""
    margin_before: int | None = None
    """Replacement context margin before a detected lane change."""
    margin_after: int | None = None
    """Replacement context margin after a detected lane change."""
    required_lane_changes: int | None = None
    """Replacement minimum lane-change count for positive scene windows."""
    negative_keep_every: int | None = None
    """Replacement negative scene-window retention interval."""
    full_config_type: type[LaneChangeConfig] = Field(
        default=LaneChangeConfig,
        init=False,
        repr=False,
    )


class ScenesConfig(ResolvedConfig):
    """Base configuration class for scene construction and temporal transforms."""

    horizon_frames: int = Field(gt=0)
    """Number of frames included in each scene horizon."""
    sample_time: float = Field(gt=0)
    """Time interval between consecutive frames in seconds."""
    window: WindowConfig | None = Field(default=None)
    """Optional sliding-window sampling configuration."""
    resample: ResampleConfig | None = Field(default=None)
    """Optional temporal resampling configuration applied before scene emission."""
    lane_change: LaneChangeConfig | None = Field(default=None)
    """Optional lane-change-aware sampling configuration."""


class ScenesPatch(ConfigPatch[ScenesConfig]):
    """Patch model for overriding scene construction settings."""

    horizon_frames: int | None = None
    """Replacement number of frames per scene horizon."""
    sample_time: float | None = None
    """Replacement frame interval in seconds."""
    window: Clearable[WindowPatch] = None
    """Patch override for sliding-window sampling settings."""
    resample: Clearable[ResamplePatch] = None
    """Patch override for temporal resampling settings."""
    lane_change: Clearable[LaneChangePatch] = None
    """Patch override for lane-change-aware sampling settings."""
    full_config_type: type[ScenesConfig] = Field(default=ScenesConfig, init=False, repr=False)

    @override
    def merge_into(self, target: ScenesConfig | None) -> ScenesConfig:
        """Apply this partial scenes config to an existing full scenes config."""
        return ScenesConfig(
            horizon_frames=_resolve_required(
                "horizon_frames",
                self.horizon_frames,
                target.horizon_frames if target is not None else None,
            ),
            sample_time=_resolve_required(
                "sample_time",
                self.sample_time,
                target.sample_time if target is not None else None,
            ),
            window=_apply_optional_block(
                self.window,
                target.window if target is not None else None,
            ),
            resample=_apply_optional_block(
                self.resample,
                target.resample if target is not None else None,
            ),
            lane_change=_apply_optional_block(
                self.lane_change,
                target.lane_change if target is not None else None,
            ),
        )


def _resolve_required(name: str, value: T | None, fallback: T | None) -> T:
    result = value if value is not None else fallback
    if result is None:
        msg = f"Missing required field: {name}"
        raise ValueError(msg)
    return result


ConfigT = TypeVar("ConfigT", bound=ConfigBase)


def _apply_optional_block(
    patch: Clearable[ConfigPatch[ConfigT]],
    target: ConfigT | None,
) -> ConfigT | None:
    """Apply a patch to an optional nested config block."""
    if patch is None:
        return target
    if isinstance(patch, Clear):
        return None
    return patch.merge_into(target)


def effective_scene_window(config: ScenesConfig) -> tuple[int, float]:
    """Return horizon frames and `sample_time` after resampling."""
    if config.resample is None:
        return config.horizon_frames, config.sample_time

    up = config.resample.up
    down = config.resample.down
    horizon_resampled = _resample_length(config.horizon_frames, up=up, down=down)
    return (horizon_resampled, config.sample_time * down / up)


def _resample_length(length: int, *, up: int, down: int) -> int:
    if length <= 0:
        return 0
    return ((length - 1) * up) // down + 1


class ScreeningConfig(ResolvedConfig):
    """Named executable cleanup, scene, and agent screening rules."""

    cleanup: dict[str, CleanupRule] = Field(default_factory=dict)
    scenes: dict[str, SceneCheckRule] = Field(default_factory=dict)
    agents: dict[str, AgentCheckRule] = Field(default_factory=dict)


class ScreeningPatch(ConfigPatch[ScreeningConfig]):
    """Patch model for named screening rule sets.

    Each rule section has independent replace/extend/remove semantics.
    This avoids deleting rules from unrelated sections that happen to share the
    same name.
    """

    cleanup: MappingPatch[CleanupRule] | None = None
    scenes: MappingPatch[SceneCheckRule] | None = None
    agents: MappingPatch[AgentCheckRule] | None = None
    full_config_type: type[ScreeningConfig] = Field(ScreeningConfig, repr=False, init=False)

    @override
    def merge_into(self, target: ScreeningConfig | None) -> ScreeningConfig:
        base = target or ScreeningConfig()
        return ScreeningConfig(
            cleanup=(
                self.cleanup.merge_into(base.cleanup) if self.cleanup is not None else base.cleanup
            ),
            scenes=self.scenes.merge_into(base.scenes) if self.scenes is not None else base.scenes,
            agents=self.agents.merge_into(base.agents) if self.agents is not None else base.agents,
        )


class PredictionTaskConfig(ResolvedConfig):
    """Forecasting bounds expressed in source-frame indices.

    Bounds use half-open indexing: history occupies `[0, prediction_origin)`
    and supervised prediction occupies `[prediction_origin, prediction_end)`.
    """

    prediction_origin: int = Field(gt=0)
    """Index of the first frame to predict."""
    prediction_end: int = Field(gt=0)
    """Exclusive end index of the supervised prediction interval."""
    require_history_endpoint: bool = True
    """Require an eligible agent at the final history frame."""

    @model_validator(mode="after")
    def _validate_order(self) -> PredictionTaskConfig:
        if self.prediction_origin >= self.prediction_end:
            msg = "`prediction_origin` must be less than `prediction_end`."
            raise ConfigurationError(msg)
        return self


class DatasetConfig(ResolvedConfig):
    """Full dataset/profile-style configuration schema."""

    scenes: ScenesConfig
    task: PredictionTaskConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    screening: ScreeningConfig | None = Field(default=None)
    output: OutputConfig = Field(default_factory=OutputConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    read: ReadConfig = Field(default_factory=ReadAll)
    assign: AssignConfig = Field(default_factory=NoAssign)
    loader_options: dict[str, Any] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_task_horizon(self) -> DatasetConfig:
        if self.task is not None and self.task.prediction_end > self.scenes.horizon_frames:
            msg = (
                "`task.prediction_end` must be less than or equal to "
                f"`scenes.horizon_frames` ({self.scenes.horizon_frames})."
            )
            raise ConfigurationError(msg)
        return self


class DatasetConfigPatchBase(ConfigBase):
    """Common optional fields shared by partial dataset-style configs.

    This base is reused by authored dataset entries and profile fragments.
    """

    scenes: ScenesPatch | None = Field(default=None)
    runtime: RuntimePatch | None = Field(default=None)
    screening: Clearable[ScreeningPatch] = None
    output: OutputPatch | None = Field(default=None)
    map: MapPatch | None = Field(default=None)
    read: ReadConfig | None = Field(default=None)
    assign: AssignConfig | None = Field(default=None)
    loader_options: Clearable[DictPatch] = None


class DatasetConfigPatch(DatasetConfigPatchBase, ConfigPatch[DatasetConfig]):
    """Patch model for applying partial values to a full dataset config.

    The merge strategy preserves existing nested defaults unless a matching
    partial model is provided.
    """

    full_config_type: type[DatasetConfig] = DatasetConfig
    task: Clearable[PredictionTaskConfig] = None
    """Complete replacement prediction task, or `clear` to remove it."""

    @override
    def merge_into(self, target: DatasetConfig | None) -> DatasetConfig:
        if target is None:
            msg = "Defaults must be provided to apply a DatasetConfigPatch."
            raise ValueError(msg)

        return DatasetConfig(
            scenes=self.scenes.merge_into(target.scenes) if self.scenes else target.scenes,
            task=_replace_prediction_task(self.task, target.task),
            runtime=self.runtime.merge_into(target.runtime) if self.runtime else target.runtime,
            screening=apply_optional(self.screening, target.screening),
            loader_options=_apply_loader_options_patch(self.loader_options, target.loader_options),
            output=self.output.merge_into(target.output) if self.output else target.output,
            map=self.map.merge_into(target.map) if self.map else target.map,
            read=self.read if self.read is not None else target.read,
            assign=self.assign if self.assign is not None else target.assign,
        )


def _replace_prediction_task(
    replacement: PredictionTaskConfig | Clear | None,
    target: PredictionTaskConfig | None,
) -> PredictionTaskConfig | None:
    if replacement is None:
        return target
    if isinstance(replacement, Clear):
        return None
    return replacement


def effective_prediction_bounds(config: DatasetConfig) -> tuple[int, int] | None:
    """Return half-open prediction bounds after optional temporal resampling."""
    task = config.task
    if task is None:
        return None
    resample = config.scenes.resample
    if resample is None:
        return task.prediction_origin, task.prediction_end

    origin = _resample_length(task.prediction_origin, up=resample.up, down=resample.down)
    end = _resample_length(task.prediction_end, up=resample.up, down=resample.down)
    if origin >= end:
        msg = "Temporal resampling collapses the configured prediction interval."
        raise ConfigurationError(msg)
    return origin, end


def _apply_loader_options_patch(
    patch: Clearable[DictPatch],
    target: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if patch is None:
        return target
    if isinstance(patch, Clear):
        return None
    return patch.merge_into(target or {})
