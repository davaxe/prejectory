"""Internal processing request and plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prejectory.config.models import (
    NoAssign,
    PreserveNativeAssign,
    ReadAll,
    ReadNative,
    SceneAssign,
    SourceAssign,
)
from prejectory.core.categories import DatasetSplit

if TYPE_CHECKING:
    from prejectory.config.models import (
        AssignConfig,
        MapConfig,
        ReadConfig,
        ScenesConfig,
        ScreeningConfig,
    )
    from prejectory.processing.loading.models import LoaderOptionsModel


def _ordered_splits(
    splits: tuple[DatasetSplit, ...] | frozenset[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    order = {DatasetSplit.TRAIN: 0, DatasetSplit.VAL: 1, DatasetSplit.TEST: 2}
    return tuple(sorted(splits, key=order.__getitem__))


@dataclass(frozen=True, slots=True)
class ReadSelection:
    """Read scope for dataset DatasetSource enumeration."""

    config: ReadConfig
    native_splits: tuple[DatasetSplit, ...] | None = None

    @classmethod
    def from_config(
        cls,
        read: ReadConfig,
        *,
        supported_native_splits: tuple[DatasetSplit, ...] | None = None,
    ) -> ReadSelection:
        """Build a read scope from the public read configuration."""
        match read:
            case ReadAll():
                return cls(config=read, native_splits=supported_native_splits)
            case ReadNative(splits=splits) if splits is not None:
                return cls(config=read, native_splits=_ordered_splits(splits))
            case ReadNative():
                return cls(config=read, native_splits=supported_native_splits)

    @property
    def strategy(self) -> str:
        """Read strategy name."""
        return self.config.strategy


@dataclass(frozen=True, slots=True)
class SplitAssignmentPlan:
    """Output split assignment plan derived from the resolved config."""

    config: AssignConfig
    seed: int | None = None

    @classmethod
    def from_config(cls, assign: AssignConfig, *, seed: int | None = None) -> SplitAssignmentPlan:
        """Build an output-assignment plan from the public config wrapper."""
        return cls(config=assign, seed=seed)

    def active(self) -> tuple[tuple[DatasetSplit, float], ...]:
        """Return active output splits and their weights."""
        if isinstance(self.config, NoAssign | PreserveNativeAssign):
            return ()
        active: list[tuple[DatasetSplit, float]] = []
        ratio = self.config.ratio
        if ratio.train > 0:
            active.append((DatasetSplit.TRAIN, ratio.train))
        if ratio.val > 0:
            active.append((DatasetSplit.VAL, ratio.val))
        if ratio.test > 0:
            active.append((DatasetSplit.TEST, ratio.test))
        return tuple(active)

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return active output split labels."""
        return tuple(split for split, _ in self.active())

    def active_weights(self) -> tuple[float, ...]:
        """Return active output split weights."""
        return tuple(weight for _, weight in self.active())

    def uses_weighted_assignment(self) -> bool:
        """Return whether the assignment uses weighted routing."""
        return isinstance(self.config, (SceneAssign, SourceAssign))

    def preserves_native_assignment(self) -> bool:
        """Return whether output assignment should preserve native labels."""
        return self.strategy == "preserve-native"

    def output_splits(
        self,
        *,
        input_native_splits: tuple[DatasetSplit, ...] | None = None,
    ) -> tuple[DatasetSplit, ...] | None:
        """Return the split directories expected for this assignment plan."""
        if isinstance(self.config, NoAssign):
            return None
        if self.preserves_native_assignment():
            return input_native_splits
        return self.active_splits()

    @property
    def gap(self) -> int | None:
        """Configured temporal assignment gap, if any."""
        return getattr(self.config, "gap", None)

    @property
    def segments(self) -> int | None:
        """Configured shuffled-time segment count, if any."""
        return getattr(self.config, "segments", None)

    @property
    def strategy(self) -> str:
        """Assignment strategy name."""
        return self.config.strategy


@dataclass(frozen=True, slots=True)
class LoaderPlan:
    """Narrow loader-facing request derived from a resolved dataset config."""

    scenes: ScenesConfig
    loader_options: LoaderOptionsModel
    read: ReadSelection
    dataset: str | None = None
    screening: ScreeningConfig | None = None
    map: MapConfig | None = None

    @property
    def horizon_frames(self) -> int:
        """Number of frames per scene horizon."""
        return self.scenes.horizon_frames


@dataclass(frozen=True, slots=True)
class TrajectoryPipelinePlan:
    """Trajectory-processing plan for one run."""

    scenes: ScenesConfig
    screening: ScreeningConfig | None = None
    assignment: SplitAssignmentPlan | None = None
