"""Scene-scoped screening rules for trajectory scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, override

import polars as pl
from pydantic import Field, field_validator, model_validator

from prejectory.core.categories import AgentCategory, AgentCategoryInput, AgentCategoryLike
from prejectory.processing.screening.base import (
    AgentCategorySelector,
    CountRange,
    FrameInput,
    FrameSet,
    RuleId,
    SceneCheckRuleBase,
    coerce_frame_set,
)

if TYPE_CHECKING:
    from prejectory.processing.screening.base import ScreeningContext


class AgentRange(SceneCheckRuleBase):
    """Require a minimum and/or maximum number of retained agents in the scene."""

    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)
    selector: AgentCategorySelector | None = None
    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)

    @model_validator(mode="after")
    def _val_range(self) -> AgentRange:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the retained-agent count range."""
        count = ctx.retained_agent_count(self.selector)
        cond = pl.lit(value=True)
        if self.minimum is not None:
            cond &= count >= self.minimum
        if self.maximum is not None:
            cond &= count <= self.maximum
        return cond


class CategoryRange(SceneCheckRuleBase):
    """Require a minimum and/or maximum number of retained agents in each category."""

    ranges: dict[AgentCategory, CountRange]
    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)

    @field_validator("ranges", mode="before")
    @classmethod
    def _coerce_categories(
        cls,
        ranges: dict[AgentCategoryLike, CountRange],
    ) -> dict[AgentCategory, CountRange]:
        return {AgentCategory.from_value(category): value for category, value in ranges.items()}

    @classmethod
    def define(cls, *ranges: tuple[AgentCategoryInput, CountRange]) -> CategoryRange:
        """Alternate constructor that accepts one or many category-range pairs."""
        return cls(ranges={AgentCategory(category): r for category, r in ranges})

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the per-category agent ranges."""
        cond = pl.lit(value=True)
        for category, r in self.ranges.items():
            count = ctx.retained_agent_count(AgentCategorySelector.include(category))
            if r.minimum is not None:
                cond &= count >= r.minimum
            if r.maximum is not None:
                cond &= count <= r.maximum
        return cond


class SceneRequireFrames(SceneCheckRuleBase):
    """Require specific relative frames to exist in the scene window."""

    frames: FrameSet
    rule: Literal["scene_frames"] = Field("scene_frames", repr=False, init=False)

    @classmethod
    def define(cls, frames: FrameInput, *, rule_id: RuleId | None = None) -> SceneRequireFrames:
        """Alternate constructor that accepts one or many frame indices."""
        return cls(frames=coerce_frame_set(frames), rule_id=rule_id)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the required frame set."""
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_scene_window(required) == len(self.frames)


class SceneRequireWindow(SceneCheckRuleBase):
    """Require a minimum fraction of frames in a relative scene window."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    rule: Literal["scene_window"] = Field("scene_window", repr=False, init=False)

    @model_validator(mode="after")
    def _validate_window(self) -> SceneRequireWindow:
        if self.end_frame < self.start_frame:
            msg = "end_frame must be greater than or equal to start_frame."
            raise ValueError(msg)
        return self

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the required window coverage."""
        relative_frame = ctx.relative_frame()
        in_window = relative_frame.is_between(self.start_frame, self.end_frame, closed="both")
        covered_frames = ctx.over_scene_window(relative_frame.filter(in_window).n_unique())
        required_coverage = round((self.end_frame - self.start_frame + 1) * self.min_fraction)
        return covered_frames >= required_coverage


class SceneMaxMissingFrames(SceneCheckRuleBase):
    """Require the cleaned scene window to stay within a missing-frame budget."""

    maximum: int = Field(ge=0)
    selector: AgentCategorySelector
    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the missing-frame budget."""
        frame = pl.col(ctx.columns.frame).filter(ctx.selector_mask(self.selector))
        unique_frame_count = ctx.over_scene_window(frame.n_unique())
        frame_span = ctx.over_scene_window(frame.max() - frame.min() + 1)
        return (frame_span - unique_frame_count) <= self.maximum


SceneCheckRule = Annotated[
    AgentRange | CategoryRange | SceneRequireFrames | SceneRequireWindow | SceneMaxMissingFrames,
    Field(discriminator="rule"),
]
"""Discriminated union of executable scene-scoped screening rule types.

These rule objects evaluate aggregate properties of a cleaned scene such as
frame coverage, retained-agent counts, or per-category count ranges.
"""

__all__ = [
    "AgentRange",
    "CategoryRange",
    "SceneCheckRule",
    "SceneCheckRuleBase",
    "SceneMaxMissingFrames",
    "SceneRequireFrames",
    "SceneRequireWindow",
]
