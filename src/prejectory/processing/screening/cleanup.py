"""Cleanup rules that prune rows or agents before scene screening."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, override

import polars as pl
from pydantic import Field, model_validator

from prejectory.core.categories import AgentCategoryInput, coerce_agent_categories
from prejectory.processing.screening.agent import AgentCheckRule  # ruff: ignore[typing-only-first-party-import]
from prejectory.processing.screening.base import AgentSet, CleanupRuleBase, RuleId

if TYPE_CHECKING:
    from prejectory.processing.screening.base import ScreeningContext


class PruneByRule(CleanupRuleBase):
    """Remove rows for agents that fail the given screening rule."""

    agent_rule: AgentCheckRule
    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)

    @model_validator(mode="after")
    def _reject_aggregate_requirements(self) -> PruneByRule:
        if self.agent_rule.require is not None:
            msg = "`require` is only valid for agent screening rules, not cleanup pruning."
            raise ValueError(msg)
        return self

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression for the wrapped agent rule."""
        scope = ctx.selector_mask(self.agent_rule.selector)
        agent_validity = self.agent_rule.predicate_expr(ctx)
        return pl.when(scope).then(agent_validity).otherwise(statement=True)


class ExcludeCategories(CleanupRuleBase):
    """Remove rows that belong to the given agent categories."""

    categories: AgentSet
    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)

    @classmethod
    def define(
        cls,
        categories: AgentCategoryInput,
        *,
        rule_id: RuleId | None = None,
    ) -> ExcludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression that excludes selected categories."""
        return ~pl.col(ctx.columns.category).is_in(self.categories)


class IncludeCategories(CleanupRuleBase):
    """Keep only rows that belong to the given agent categories."""

    categories: AgentSet
    rule: Literal["include"] = Field("include", repr=False, init=False)

    @classmethod
    def define(
        cls,
        categories: AgentCategoryInput,
        *,
        rule_id: RuleId | None = None,
    ) -> IncludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression that keeps selected categories."""
        return pl.col(ctx.columns.category).is_in(self.categories)


type CleanupRule = Annotated[
    PruneByRule | ExcludeCategories | IncludeCategories,
    Field(discriminator="rule"),
]
"""Discriminated union of executable cleanup-rule types.

Cleanup rules run before screening checks and determine which scene rows are
retained for subsequent scene- and agent-level evaluation.
"""

__all__ = [
    "CleanupRule",
    "CleanupRuleBase",
    "ExcludeCategories",
    "IncludeCategories",
    "PruneByRule",
]
