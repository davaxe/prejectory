"""Agent-scoped screening rules for trajectory scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, override

import polars as pl
from pydantic import Field, model_validator

from prejectory.processing.screening.base import (
    AgentCategorySelector,
    AgentCheckRuleBase,
    FrameInput,
    FrameSet,
    PassingRequirement,
    RuleId,
    Tolerance,
    coerce_frame_set,
)

if TYPE_CHECKING:
    from prejectory.processing.screening.base import ScreeningContext


class MinDistance(AgentCheckRuleBase):
    """Require a minimum total distance traveled per agent."""

    rule: Literal["min_distance"] = Field("min_distance", repr=False, init=False)
    minimum: float = Field(gt=0)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        x, y = pl.col(ctx.columns.require("x")), pl.col(ctx.columns.require("y"))
        dx, dy = x - x.shift(1), y - y.shift(1)
        step_distance = (dx.pow(2) + dy.pow(2)).sqrt()
        return ctx.over_agent_window(step_distance.sum()) >= self.minimum


class AgentRequireFrames(AgentCheckRuleBase):
    """Require each retained agent to cover specific relative frames."""

    rule: Literal["frames"] = Field("frames", repr=False, init=False)
    frames: FrameSet

    @classmethod
    def define(
        cls,
        frames: FrameInput,
        *,
        selector: AgentCategorySelector | None = None,
        tolerance: Tolerance | None = None,
        rule_id: RuleId | None = None,
        require: PassingRequirement | None = None,
    ) -> AgentRequireFrames:
        """Return a defined rule."""
        return cls(
            frames=coerce_frame_set(frames),
            selector=selector,
            tolerance=tolerance,
            rule_id=rule_id,
            require=require,
        )

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the configured frame set."""
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_agent_window(required) == len(self.frames)


class AgentRequireWindow(AgentCheckRuleBase):
    """Require a minimum fraction of frames in a relative agent window."""

    rule: Literal["window"] = Field("window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_window(self) -> AgentRequireWindow:
        if self.end_frame < self.start_frame:
            msg = "end_frame must be greater than or equal to start_frame."
            raise ValueError(msg)
        return self

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the configured window."""
        relative_frame = ctx.relative_frame()
        in_window = relative_frame.is_between(self.start_frame, self.end_frame, closed="both")
        covered_frames = ctx.over_agent_window(relative_frame.filter(in_window).n_unique())
        required_coverage = (self.end_frame - self.start_frame + 1) * self.min_fraction
        return covered_frames.cast(pl.Float64) >= required_coverage


class MinObservations(AgentCheckRuleBase):
    """Require a minimum number of observations per agent."""

    rule: Literal["min_observations"] = Field("min_observations", repr=False, init=False)
    minimum: int = Field(ge=1)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the minimum observation count."""
        return ctx.over_agent_window(pl.len()) >= self.minimum


class AgentMaxMissingFrames(AgentCheckRuleBase):
    """Require a maximum number of missing frames per agent."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    maximum: int = Field(ge=0)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the missing-frame budget."""
        scene_length = ctx.scene_length()
        agent_length = ctx.over_agent_window(pl.col(ctx.columns.frame).n_unique())
        missing_frames = scene_length - agent_length
        return missing_frames <= self.maximum


class MaxGap(AgentCheckRuleBase):
    """Require the gap between two frames to be less than or equal to a maximum."""

    rule: Literal["max_gap"] = Field("max_gap", repr=False, init=False)
    maximum: int = Field(ge=0)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the maximum frame gap."""
        frame_col = pl.col(ctx.columns.frame)
        gaps = frame_col - frame_col.shift(1) - 1
        max_gap = ctx.over_agent_window(gaps.max())
        return max_gap <= self.maximum


class MinConsecutiveFrames(AgentCheckRuleBase):
    """Require a minimum longest consecutive run per agent."""

    rule: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames",
        repr=False,
        init=False,
    )
    minimum: int = Field(ge=1)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the longest consecutive run."""
        frame_col = pl.col(ctx.columns.frame)
        prev_frame = frame_col.shift(1)
        frame_diff = frame_col - prev_frame
        new_run = prev_frame.is_null() | (frame_diff != 1)
        row_idx = ctx.over_agent_window(frame_col.cum_count())
        run_start_idx = ctx.over_agent_window(
            pl.when(new_run).then(row_idx).otherwise(None).forward_fill(),
        )
        streak_len = row_idx - run_start_idx + 1
        longest_run = ctx.over_agent_window(streak_len.max())
        return longest_run >= self.minimum


class StartsByFrame(AgentCheckRuleBase):
    """Require each retained agent to start by a specific relative frame."""

    frame: int = Field(ge=0)
    rule: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the first observed frame."""
        first_frame = ctx.over_agent_window(pl.col(ctx.columns.frame).min())
        return first_frame <= self.frame


class EndsAfterFrame(AgentCheckRuleBase):
    """Require each retained agent to end after a specific relative frame."""

    rule: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)
    frame: int = Field(ge=0)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the last observed frame."""
        last_frame = ctx.over_agent_window(pl.col(ctx.columns.frame).max())
        return last_frame >= self.frame


class MinSpan(AgentCheckRuleBase):
    """Require a minimum span between the first and last frame of each retained agent."""

    rule: Literal["min_span"] = Field("min_span", repr=False, init=False)
    minimum: int = Field(ge=1)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the per-agent pass expression for the minimum frame span."""
        first_frame = ctx.over_agent_window(pl.col(ctx.columns.frame).min())
        last_frame = ctx.over_agent_window(pl.col(ctx.columns.frame).max())
        span = last_frame - first_frame + 1
        return span >= self.minimum


def invalid_agent_tolerance_expr(
    tolerance: Tolerance | None,
    *,
    invalid_agents: pl.Expr,
    invalid_fraction: pl.Expr,
) -> pl.Expr:
    """Return the scene-pass expression for a configured invalid-agent tolerance."""
    if tolerance is None:
        return invalid_agents == 0
    if tolerance.relative is not None and tolerance.absolute is None:
        return invalid_fraction <= tolerance.relative
    if tolerance.absolute is not None and tolerance.relative is None:
        return invalid_agents <= tolerance.absolute
    return pl.all_horizontal(
        invalid_agents <= tolerance.absolute,
        invalid_fraction <= tolerance.relative,
    )


def passing_requirement_expr(
    requirement: PassingRequirement | None,
    *,
    passing_agents: pl.Expr,
    passing_fraction: pl.Expr,
) -> pl.Expr:
    """Return the scene-pass expression for configured passing-agent requirements."""
    if requirement is None:
        return pl.lit(value=True)
    if requirement.relative is not None and requirement.absolute is None:
        return passing_fraction >= requirement.relative
    if requirement.absolute is not None and requirement.relative is None:
        return passing_agents >= requirement.absolute
    return pl.all_horizontal(
        passing_agents >= requirement.absolute,
        passing_fraction >= requirement.relative,
    )


type AgentCheckRule = Annotated[
    MinDistance
    | AgentMaxMissingFrames
    | AgentRequireFrames
    | AgentRequireWindow
    | MinObservations
    | MaxGap
    | MinConsecutiveFrames
    | StartsByFrame
    | EndsAfterFrame
    | MinSpan,
    Field(discriminator="rule"),
]
"""Discriminated union of executable agent-scoped screening rule types.

Each variant encapsulates the Polars expression needed to evaluate one
per-agent screening rule against a screening context.
"""

__all__ = [
    "AgentCheckRule",
    "AgentCheckRuleBase",
    "AgentMaxMissingFrames",
    "AgentRequireFrames",
    "AgentRequireWindow",
    "EndsAfterFrame",
    "MaxGap",
    "MinConsecutiveFrames",
    "MinDistance",
    "MinObservations",
    "MinSpan",
    "StartsByFrame",
]
