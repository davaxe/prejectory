# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
"""Runtime DatasetSource processing and scene materialization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from prejectory.core.categories import DatasetSplit
from prejectory.core.errors import SplitAssignmentError
from prejectory.core.scene import Scene
from prejectory.processing.columns import TrajectoryColumns
from prejectory.processing.loading.assigner import StatelessWeightedAssigner
from prejectory.processing.maps import MapReference, bind_map_provider
from prejectory.processing.models import SplitAssignmentPlan, TrajectoryPipelinePlan
from prejectory.processing.trajectory import build_trajectory_processing_stages
from prejectory.runtime.types import trajectory_schema_after_transforms

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from prejectory.core.scene import TrajectorySchema
    from prejectory.core.scene.model import MapResolver
    from prejectory.processing.loading.base import SceneLoader
    from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame
    from prejectory.processing.screening.screen import CleanupSceneStats
    from prejectory.processing.trajectory import TrajectoryProcessingStages
    from prejectory.runtime.types import ExecutionPlan


@dataclass(frozen=True, slots=True)
class SceneIdentifier:
    """Stable identifier for one candidate scene within a DatasetSource."""

    source_identifier: object
    source_local_scene_index: int


@dataclass(frozen=True, slots=True)
class SceneCandidate:
    """Runtime scene candidate produced from one processed DatasetSource frame."""

    source: DatasetSource[Any]
    stable_identifier: SceneIdentifier
    frame: pl.DataFrame
    passes_screening: bool
    cleanup_stats: CleanupSceneStats | None = None
    map_reference: MapReference = field(default_factory=MapReference)
    passed_agent_ids: frozenset[int] | None = None
    split_assignment: DatasetSplit | None = None
    ego_agent_id: int | None = None


class SplitAssigner:
    """Assign output splits to selected scene candidates."""

    def __init__(self, request: SplitAssignmentPlan | None) -> None:
        self.request: SplitAssignmentPlan | None = request
        self._weighted_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

        if request is not None and request.uses_weighted_assignment():
            self._weighted_assigner = StatelessWeightedAssigner(
                request.active_splits(),
                request.active_weights(),
                seed=request.seed,
            )

    def assign(
        self,
        *,
        source: DatasetSource[Any],
        stable_identifier: SceneIdentifier,
        frame: pl.DataFrame,
    ) -> DatasetSplit | None:
        """Assign split to scene candidate."""
        if self.request is None:
            return None

        match self.request.strategy:
            case "time" | "shuffled-time":
                return self._resolve_time_split(frame)
            case "scene":
                return self._resolve_scene_split(source, stable_identifier)
            case "source":
                return self._resolve_source_split(source)
            case "preserve-native":
                return source.source_split
            case _:
                return None

    @staticmethod
    def _resolve_time_split(frame: pl.DataFrame) -> DatasetSplit:
        if "split" not in frame:
            msg = "Did not get split column in dataframe."
            raise SplitAssignmentError(msg)

        split_str = str(frame["split"].first()).lower()
        try:
            return DatasetSplit(split_str)
        except ValueError as exc:
            msg = f"Invalid split assignment '{split_str}' in dataframe."
            raise SplitAssignmentError(msg) from exc

    def _resolve_scene_split(
        self,
        source: DatasetSource[Any],
        stable_identifier: SceneIdentifier,
    ) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None

        return self._weighted_assigner.assign(
            stable_identifier.source_local_scene_index,
            str(source.identifier),
        )

    def _resolve_source_split(self, source: DatasetSource[Any]) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None

        return self._weighted_assigner.assign(str(source.identifier))


@dataclass(slots=True)
class RuntimeProcessor:
    """Own the full runtime DatasetSource-to-scene processing flow for one plan."""

    dataset: str
    loader: SceneLoader[Any, Any]
    source_schema: TrajectorySchema
    target_schema: TrajectorySchema
    horizon_frames: int
    sample_time: float
    split_assigner: SplitAssigner
    _processing_stages: TrajectoryProcessingStages | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def from_plan(
        cls,
        plan: ExecutionPlan,
        loader: SceneLoader[Any, Any],
    ) -> RuntimeProcessor:
        """Create processor from execution plan and scene loader."""
        return cls(
            dataset=plan.dataset,
            loader=loader,
            source_schema=trajectory_schema_after_transforms(
                plan._descriptor.native_schema,
                plan.config,
            ),
            target_schema=plan.trajectory_schema,
            horizon_frames=plan.effective_horizon_frames,
            sample_time=plan.effective_sample_time,
            split_assigner=SplitAssigner(plan._assignment),
        )

    def iter_sources(self) -> Iterable[DatasetSource[Any]]:
        """Iterate all DatasetSources to process."""
        yield from self.loader.iter_sources()

    def total_sources(self) -> int | None:
        """Return total number of sources if known."""
        return self.loader.count_sources()

    def screening_enabled(self) -> bool:
        """Return whether screening is enabled for this processor."""
        config = self.loader.screening_config
        return config is not None and bool(config.cleanup or config.agents or config.scenes)

    def iter_candidates(self, source: DatasetSource[Any]) -> Iterable[SceneCandidate]:
        """Iterate scene candidates for one DatasetSource."""
        source_local_scene_index = 0

        for data in self.loader.load_source(source):
            effective_source = self._effective_source(source, data)
            stages = self._processing_stages_for_loader()

            for candidate_frame in stages.iter_prescreen_frames(data.frame):
                stable_identifier = SceneIdentifier(
                    source_identifier=effective_source.identifier,
                    source_local_scene_index=source_local_scene_index,
                )
                screening = stages.screen(candidate_frame)

                if not screening.passes_scene:
                    yield SceneCandidate(
                        source=effective_source,
                        stable_identifier=stable_identifier,
                        frame=screening.frame,
                        passes_screening=False,
                        cleanup_stats=screening.cleanup,
                        map_reference=data.map_reference,
                        passed_agent_ids=screening.passed_agent_ids,
                        ego_agent_id=data.ego_agent_id,
                    )
                else:
                    for output_frame in stages.iter_output_frames(screening.frame):
                        split_assignment = self.split_assigner.assign(
                            source=effective_source,
                            stable_identifier=stable_identifier,
                            frame=output_frame,
                        )
                        yield SceneCandidate(
                            source=effective_source,
                            stable_identifier=stable_identifier,
                            frame=output_frame,
                            passes_screening=True,
                            cleanup_stats=screening.cleanup,
                            map_reference=data.map_reference,
                            passed_agent_ids=screening.passed_agent_ids,
                            split_assignment=split_assignment,
                            ego_agent_id=data.ego_agent_id,
                        )
                source_local_scene_index += 1

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        """Materialize one scene from candidate."""
        map_key, map_resolver = self._resolve_scene_map(candidate)
        scene = Scene.create(
            frame=candidate.frame,
            scene_number=scene_number,
            horizon_frames=self.horizon_frames,
            schema=self.source_schema,
            sample_time=self.sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            passed_agent_ids=candidate.passed_agent_ids,
            split_assignment=candidate.split_assignment,
            dataset=self.dataset,
            ego_agent_id=candidate.ego_agent_id,
        )

        if self.target_schema == self.source_schema:
            return scene

        return scene.as_schema(self.target_schema)

    def _processing_stages_for_loader(self) -> TrajectoryProcessingStages:
        if self._processing_stages is not None:
            return self._processing_stages

        plan = TrajectoryPipelinePlan(
            scenes=self.loader.scenes_config,
            screening=self.loader.screening_config,
            assignment=self.split_assigner.request,
        )
        columns = TrajectoryColumns.from_schema(self.loader.native_trajectory_schema())

        self._processing_stages = build_trajectory_processing_stages(plan, columns=columns)
        return self._processing_stages

    def _resolve_scene_map(
        self,
        candidate: SceneCandidate,
    ) -> tuple[str | None, MapResolver | None]:
        if self.loader.map_config is None:
            return None, None

        map_key = candidate.map_reference.map_key or candidate.source.map_key
        reference = candidate.map_reference

        if reference.map_key != map_key:
            reference = replace(reference, map_key=map_key)

        provider = self.loader.map_provider
        if provider is None:
            return map_key, None

        return map_key, bind_map_provider(provider, reference)

    @staticmethod
    def _effective_source(
        source: DatasetSource[Any],
        data: LoadedSourceFrame,
    ) -> DatasetSource[Any]:
        if data.source_split is None:
            return source

        return source.with_source_split(data.source_split)
