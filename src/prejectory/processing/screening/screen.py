"""ScreeningRuleSet containers, declarative configs, and application helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import polars as pl

from prejectory.core.functional.basic import normalize_group_by
from prejectory.processing.screening.agent import (
    AgentCheckRule,
    invalid_agent_tolerance_expr,
    passing_requirement_expr,
)
from prejectory.processing.screening.base import (
    AgentCheckRuleBase,
    Rule,
    ScreeningContext,
    rule_name,
)
from prejectory.processing.screening.cleanup import CleanupRule, PruneByRule
from prejectory.processing.screening.scene import SceneCheckRule  # ruff: ignore[typing-only-first-party-import]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from prejectory.config.models import ScreeningConfig
    from prejectory.processing.columns import TrajectoryColumns
    from prejectory.processing.screening.base import CleanupRuleBase, SceneCheckRuleBase


RELATIVE_FRAME_COLUMN: Final[str] = "_screening_relative_frame"


@dataclass(frozen=True, slots=True)
class CleanupRuleSceneStats:
    """Cleanup statistics for one rule within one candidate scene."""

    rule_name: str
    rows_before: int
    rows_after: int
    agents_before: int
    agents_after: int

    @property
    def rows_removed(self) -> int:
        """Number of rows removed by this cleanup rule."""
        return self.rows_before - self.rows_after

    @property
    def agents_removed(self) -> int:
        """Number of agents fully removed by this cleanup rule."""
        return self.agents_before - self.agents_after


@dataclass(frozen=True, slots=True)
class CleanupSceneStats:
    """Cleanup statistics for one candidate scene across all cleanup rules."""

    rows_before: int
    rows_after: int
    agents_before: int
    agents_after: int
    by_rule: tuple[CleanupRuleSceneStats, ...] = ()

    @property
    def rows_removed(self) -> int:
        """Number of rows removed across all cleanup rules."""
        return self.rows_before - self.rows_after

    @property
    def agents_removed(self) -> int:
        """Number of agents fully removed across all cleanup rules."""
        return self.agents_before - self.agents_after


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """Structured result from screening one candidate scene."""

    frame: pl.DataFrame
    passes_scene: bool
    cleanup: CleanupSceneStats | None = None
    passed_agent_ids: frozenset[int] | None = None


@dataclass(slots=True, frozen=True)
class ScreeningRuleSet:
    """Collection of cleanup and check rules used during screening."""

    cleanup_rules: tuple[CleanupRule, ...] = ()
    scene_rules: tuple[SceneCheckRule, ...] = ()
    agent_rules: tuple[AgentCheckRule, ...] = ()

    def __post_init__(self) -> None:
        """Validate that all rules have unique names."""
        seen: set[str] = set()
        for rule in (*self.cleanup_rules, *self.scene_rules, *self.agent_rules):
            name = rule.name()
            if name in seen:
                msg = f"Duplicate rule name: {name}"
                raise ValueError(msg)
            seen.add(name)

    @classmethod
    def define(
        cls,
        cleanup_rules: Iterable[CleanupRule | AgentCheckRule] = (),
        scene_rules: Iterable[SceneCheckRule] = (),
        agent_rules: Iterable[AgentCheckRule] = (),
    ) -> ScreeningRuleSet:
        """Return a new ScreeningRuleSet instance with the given rules, validating uniqueness."""
        rules: list[CleanupRule] = []
        for rule in cleanup_rules:
            if isinstance(rule, AgentCheckRuleBase):
                rules.append(PruneByRule(agent_rule=rule))  # ty: ignore[invalid-argument-type]
            else:
                rules.append(rule)
        return cls(
            cleanup_rules=tuple(rules),
            scene_rules=tuple(scene_rules),
            agent_rules=tuple(agent_rules),
        )

    @classmethod
    def from_config(cls, config: ScreeningConfig) -> ScreeningRuleSet:
        """Return a ScreeningRuleSet instance compiled from a ScreeningConfig."""
        return cls.define(
            cleanup_rules=_named_rules(config.cleanup),
            scene_rules=_named_rules(config.scenes),
            agent_rules=_named_rules(config.agents),
        )


def _named_rules[RuleT: Rule](entries: dict[str, RuleT]) -> tuple[RuleT, ...]:
    return tuple(
        rule.model_copy(update={"rule_id": name, "enabled": True}) for name, rule in entries.items()
    )


def screen_data(
    data: pl.DataFrame,
    scene_screening: ScreeningRuleSet | None,
    columns: TrajectoryColumns,
) -> ScreeningResult:
    """Apply screening to one candidate scene and return structured metadata."""
    if scene_screening is None:
        return ScreeningResult(frame=data, passes_scene=True)

    frame, ctx, relative_frame_column = _prepare_screening_frame(data, columns)
    frame, cleanup_stats = _apply_cleanup_result(frame, scene_screening.cleanup_rules, ctx)
    scene_passes = _evaluate_scene_rules(frame, scene_screening.scene_rules, ctx)
    agent_passes, passed_agent_ids = _evaluate_agent_rules(frame, scene_screening.agent_rules, ctx)
    passes_scene = bool(frame.height > 0 and all((*scene_passes, *agent_passes)))
    frame = frame.drop(relative_frame_column, strict=False)
    return ScreeningResult(
        frame=frame,
        passes_scene=passes_scene,
        cleanup=cleanup_stats,
        passed_agent_ids=passed_agent_ids,
    )


def _prepare_screening_frame(
    data: pl.DataFrame,
    columns: TrajectoryColumns,
) -> tuple[pl.DataFrame, ScreeningContext, str]:
    ctx = _build_context(columns=columns, scene_group_by=None)
    relative_frame_column = _temporary_column_name(data, RELATIVE_FRAME_COLUMN)
    frame = data.with_columns(ctx.relative_frame().alias(relative_frame_column))
    return frame, replace(ctx, relative_frame_column=relative_frame_column), relative_frame_column


def _apply_cleanup_result(
    data: pl.DataFrame,
    rules: tuple[CleanupRuleBase, ...],
    ctx: ScreeningContext,
) -> tuple[pl.DataFrame, CleanupSceneStats | None]:
    if not rules:
        return data, None

    frame = data
    by_rule: list[CleanupRuleSceneStats] = []
    rows_before_all = frame.height
    agents_before_all = _agent_count(frame, ctx)

    for rule in rules:
        rows_before = frame.height
        agents_before = _agent_count(frame, ctx)
        frame = _filter_cleanup_rule(frame, rule, ctx)
        by_rule.append(
            CleanupRuleSceneStats(
                rule_name=rule_name(rule),
                rows_before=rows_before,
                rows_after=frame.height,
                agents_before=agents_before,
                agents_after=_agent_count(frame, ctx),
            ),
        )

    return frame, CleanupSceneStats(
        rows_before=rows_before_all,
        rows_after=frame.height,
        agents_before=agents_before_all,
        agents_after=_agent_count(frame, ctx),
        by_rule=tuple(by_rule),
    )


def _filter_cleanup_rule(
    frame: pl.DataFrame,
    rule: CleanupRuleBase,
    ctx: ScreeningContext,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    mask = (
        frame
        .lazy()
        .select(rule.predicate_expr(ctx).fill_null(value=False).alias("_cleanup_keep"))
        .collect()
        .get_column("_cleanup_keep")
    )
    return frame.filter(mask)


def _evaluate_scene_rules(
    frame: pl.DataFrame,
    rules: tuple[SceneCheckRuleBase, ...],
    ctx: ScreeningContext,
) -> tuple[bool, ...]:
    if frame.is_empty():
        return tuple(False for _ in rules)
    return tuple(_first_bool(frame, rule.predicate_expr(ctx)) for rule in rules)


def _evaluate_agent_rules(
    frame: pl.DataFrame,
    rules: tuple[AgentCheckRuleBase, ...],
    ctx: ScreeningContext,
) -> tuple[tuple[bool, ...], frozenset[int] | None]:
    if frame.is_empty():
        return tuple(False for _ in rules), frozenset()
    if not rules:
        return (), _agent_ids(frame, ctx)

    agent_pass_exprs: list[pl.Expr] = []
    scene_passes: list[bool] = []
    for rule in rules:
        agent_pass, scene_pass = _agent_rule_exprs(rule, ctx)
        agent_pass_exprs.append(agent_pass)
        scene_passes.append(_first_bool(frame, scene_pass))

    agent_rule_valid_column = "_prejectory_agent_rule_valid"
    passed_agent_column = "_prejectory_passed_agent"
    passed_frame = (
        frame
        .lazy()
        .with_columns(_and_all(agent_pass_exprs).alias(agent_rule_valid_column))
        .with_columns(
            ctx.over_agent_window(pl.col(agent_rule_valid_column).any()).alias(passed_agent_column),
        )
        .select(pl.col(ctx.columns.agent_id), pl.col(passed_agent_column))
        .collect()
        .filter(pl.col(passed_agent_column))
    )
    passed_agent_ids = frozenset(
        int(agent_id) for agent_id in passed_frame.get_column(ctx.columns.agent_id).unique()
    )
    return tuple(scene_passes), passed_agent_ids


def _first_bool(frame: pl.DataFrame, expr: pl.Expr) -> bool:
    result = frame.lazy().select(expr.fill_null(value=False).alias("_result")).collect()
    if result.is_empty():
        return False
    return bool(result.get_column("_result").first())


def _agent_count(frame: pl.DataFrame, ctx: ScreeningContext) -> int:
    if frame.is_empty():
        return 0
    return int(frame.get_column(ctx.columns.agent_id).n_unique())


def _agent_ids(frame: pl.DataFrame, ctx: ScreeningContext) -> frozenset[int]:
    return frozenset(int(agent_id) for agent_id in frame.get_column(ctx.columns.agent_id).unique())


def _build_context(
    *,
    columns: TrajectoryColumns,
    scene_group_by: str | Sequence[str] | None,
    relative_frame_column: str | None = None,
) -> ScreeningContext:
    scene_window = list(normalize_group_by(scene_group_by))
    agent_window = [*scene_window, columns.agent_id] if scene_window else [columns.agent_id]
    return ScreeningContext(
        columns=columns,
        scene_window=tuple(scene_window),
        agent_window=tuple(agent_window),
        relative_frame_column=relative_frame_column,
    )


def _temporary_column_name[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT, base_name: str
) -> str:
    schema_names = (
        data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.schema.names()
    )
    if base_name not in schema_names:
        return base_name
    index = 1
    while f"{base_name}_{index}" in schema_names:
        index += 1
    return f"{base_name}_{index}"


def _agent_rule_exprs(rule: AgentCheckRuleBase, ctx: ScreeningContext) -> tuple[pl.Expr, pl.Expr]:
    scope = ctx.selector_mask(rule.selector)
    scoped_agent_count = ctx.retained_agent_count(rule.selector)

    agent_pass = pl.when(scope).then(rule.predicate_expr(ctx)).otherwise(pl.lit(value=True))
    invalid_agents = ctx.over_scene_window(
        pl.col(ctx.columns.agent_id).filter(scope & ~agent_pass).n_unique(),
    )
    invalid_fraction = (
        pl.when(scoped_agent_count > 0).then(invalid_agents / scoped_agent_count).otherwise(0.0)
    )
    passing_agents = ctx.over_scene_window(
        pl.col(ctx.columns.agent_id).filter(scope & agent_pass).n_unique(),
    )
    passing_fraction = (
        pl.when(scoped_agent_count > 0).then(passing_agents / scoped_agent_count).otherwise(0.0)
    )
    tolerance_expr = (
        pl.lit(value=True)
        if rule.tolerance is None and rule.require is not None
        else invalid_agent_tolerance_expr(
            rule.tolerance,
            invalid_agents=invalid_agents,
            invalid_fraction=invalid_fraction,
        )
    )
    tolerance_pass = (
        pl.when(scoped_agent_count > 0).then(tolerance_expr).otherwise(pl.lit(value=True))
    )
    scene_pass = pl.all_horizontal(
        tolerance_pass,
        passing_requirement_expr(
            rule.require,
            passing_agents=passing_agents,
            passing_fraction=passing_fraction,
        ),
    )
    return agent_pass, scene_pass


def _and_all(exprs: list[pl.Expr]) -> pl.Expr:
    return pl.lit(value=True) if not exprs else pl.all_horizontal(*exprs)
