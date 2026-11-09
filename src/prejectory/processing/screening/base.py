"""Base context, selectors, and rule models for scene-screening definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, override

import polars as pl
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from prejectory.core.categories import AgentCategory, AgentCategoryInput, coerce_agent_categories

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prejectory.processing.columns import TrajectoryColumns


def coerce_frame_set(value: FrameInput) -> frozenset[int]:
    """Convert one or many frame indices into a normalized integer set."""
    values = [value] if isinstance(value, int) else value
    return frozenset(int(item) for item in values)


type FrameInput = int | Iterable[int]
"""One or more frame indices relative to the start of the scene."""
type FrameSet = Annotated[frozenset[int], Field(min_length=1), BeforeValidator(coerce_frame_set)]
"""Set of frame indices relative to the start of the scene."""
type AgentSet = Annotated[
    frozenset[AgentCategory],
    Field(min_length=1),
    BeforeValidator(lambda v: coerce_agent_categories(v, frozenset)),
]
"""Set of agent categories."""


class ScreeningModel(BaseModel):
    """Shared immutable model configuration for screening definitions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class AgentCategorySelector(ScreeningModel):
    """Restrict a rule to agents matching category predicates."""

    mode: Literal["include", "exclude"] = "include"
    categories: AgentSet

    @classmethod
    def define(
        cls,
        mode: Literal["include", "exclude"],
        categories: AgentCategoryInput,
    ) -> AgentCategorySelector:
        """Create a selector from flexible category inputs."""
        return cls(mode=mode, categories=coerce_agent_categories(categories, frozenset))

    @classmethod
    def include(cls, categories: AgentCategoryInput) -> AgentCategorySelector:
        """Create a selector that keeps only the given categories in scope."""
        return cls.define("include", categories)

    @classmethod
    def exclude(cls, categories: AgentCategoryInput) -> AgentCategorySelector:
        """Create a selector that excludes the given categories from scope."""
        return cls.define("exclude", categories)


class Tolerance(ScreeningModel):
    """Optional tolerance thresholds for relaxed agent-rule checks."""

    absolute: float | None = Field(default=None, gt=0.0)
    relative: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_tolerance(self) -> Tolerance:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class PassingRequirement(ScreeningModel):
    """Minimum selected-agent pass thresholds for scene acceptance."""

    absolute: int | None = Field(default=None, ge=1)
    relative: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_threshold(self) -> PassingRequirement:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class CountRange(ScreeningModel):
    """Inclusive minimum/maximum integer range."""

    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> CountRange:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self


@dataclass(slots=True, frozen=True)
class ScreeningContext:
    """Shared expressions and column metadata used while evaluating screening logic."""

    columns: TrajectoryColumns
    scene_window: tuple[str, ...]
    agent_window: tuple[str, ...]
    relative_frame_column: str | None = None

    def over_scene_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the scene window to an expression."""
        if not self.scene_window:
            return expr
        return expr.over(self.scene_window)

    def over_agent_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the agent window to an expression."""
        if not self.agent_window:
            return expr
        return expr.over(self.agent_window)

    def relative_frame(self) -> pl.Expr:
        """Return the scene-relative frame index expression."""
        if self.relative_frame_column is not None:
            return pl.col(self.relative_frame_column)
        return pl.col(self.columns.frame) - self.scene_start_frame()

    def scene_length(self) -> pl.Expr:
        """Return the number of frames in the current scene."""
        return self.scene_end_frame() - self.scene_start_frame() + 1

    def scene_start_frame(self) -> pl.Expr:
        """Return the starting frame of the current scene."""
        return self.over_scene_window(pl.col(self.columns.frame).min())

    def scene_end_frame(self) -> pl.Expr:
        """Return the ending frame of the current scene."""
        return self.over_scene_window(pl.col(self.columns.frame).max())

    def selector_mask(self, selector: AgentCategorySelector | None) -> pl.Expr:
        """Return a row mask for the given selector."""
        if selector is None:
            return pl.lit(value=True)

        in_scope = pl.col(self.columns.category).is_in(selector.categories)
        return in_scope if selector.mode == "include" else ~in_scope

    def retained_agent_count(self, selector: AgentCategorySelector | None = None) -> pl.Expr:
        """Return the number of retained agents in the current scene."""
        return self.over_scene_window(
            pl.col(self.columns.agent_id).filter(self.selector_mask(selector)).n_unique(),
        )


type RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]


class Rule(ScreeningModel, ABC):
    """Base class shared by all screening rules."""

    enabled: bool = Field(default=True, repr=False)
    rule_id: RuleId | None = Field(default=None, repr=False)

    @abstractmethod
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in diagnostics and merging."""
        return rule_name(self)


class CheckRuleBase(Rule, ABC):
    """Base class for rules that contribute pass/fail checks."""


class AgentCheckRuleBase(CheckRuleBase, ABC):
    """Base class for per-agent check rules."""

    selector: AgentCategorySelector | None = None
    tolerance: Tolerance | None = Field(default=None)
    require: PassingRequirement | None = Field(default=None)

    @override
    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        """Omit selector from repr when none."""
        for name, value in super().__repr_args__():
            if name == "selector" and value is None:
                continue
            yield name, value


class SceneCheckRuleBase(CheckRuleBase, ABC):
    """Base class for scene-level check rules."""


class CleanupRuleBase(Rule, ABC):
    """Base class for rules that physically remove rows before screening."""


def rule_group(rule: Rule) -> str:
    """Return the rule group used to namespace implicit rule identifiers."""
    if isinstance(rule, AgentCheckRuleBase):
        return "agent"
    if isinstance(rule, SceneCheckRuleBase):
        return "scene"
    if isinstance(rule, CleanupRuleBase):
        return "cleanup"
    return "rule"


def rule_name(rule: Rule) -> str:
    """Return the stable rule identifier used in diagnostics and merging."""
    if rule.rule_id is not None:
        return rule.rule_id
    name = getattr(rule, "rule", None)
    if not isinstance(name, str):
        msg = f"Rule {type(rule).__name__} is missing a string 'rule' discriminator."
        raise TypeError(msg)
    return f"{rule_group(rule)}_{name}"
