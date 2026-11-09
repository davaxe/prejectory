"""Planning and execution helpers for scene-field derivations.

The derivation logic for the schema conversion is designed in a way that adding
new derivation rules is straightforward and does not require changes to the core
derivation logic, which at first glance might seem overly complex but helps
keep the code maintainable and extensible as we add more fields and rules.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

import polars as pl

from prejectory.core.errors import TrajectorySchemaError
from prejectory.core.functional import derivative, yaw_from_position, yaw_from_velocity
from prejectory.core.scene.schema import TrajectoryField

_POSITION_FIELDS: Final[TrajectoryField] = TrajectoryField.X | TrajectoryField.Y
_VELOCITY_FIELDS: Final[TrajectoryField] = TrajectoryField.VX | TrajectoryField.VY
_ACCELERATION_FIELDS: Final[TrajectoryField] = TrajectoryField.AX | TrajectoryField.AY
_YAW_FIELDS: Final[TrajectoryField] = TrajectoryField.YAW
_KINEMATIC_FIELDS: Final[TrajectoryField] = _VELOCITY_FIELDS | _ACCELERATION_FIELDS

_DERIVATIVE_RENAME: Final[dict[int, list[str]]] = {1: ["vx", "vy"], 2: ["ax", "ay"]}


@dataclass(slots=True, frozen=True)
class ConversionContext:
    """Runtime information needed to derive scene fields."""

    sample_time: float | None
    group_by: str | tuple[str] | None = "id"


type DerivationApply = Callable[[pl.LazyFrame, ConversionContext], pl.LazyFrame]


@dataclass(slots=True, frozen=True)
class DerivationRule:
    """A rule that derives one or more semantic scene fields."""

    name: str
    """Unique name for this rule."""
    requires: TrajectoryField
    """Fields required to apply this rule."""
    outputs: TrajectoryField
    """Fields added by applying this rule."""
    cost: int
    """Relative cost of applying this rule, used for derivation planning."""
    apply: DerivationApply
    """Function that applies the rule to a dataframe."""
    needs_sample_time: bool = False
    """Flag to indicate if the rule requires `sample_time`."""


def apply_derivation_plan(
    data: pl.DataFrame,
    plan: Iterable[DerivationRule],
    context: ConversionContext,
    input_fields: TrajectoryField,
) -> tuple[pl.DataFrame, TrajectoryField]:
    """Apply a derivation plan in order."""
    lf = data.lazy()
    output_fields = input_fields
    for rule in plan:
        lf = rule.apply(lf, context)
        output_fields |= rule.outputs

    data = lf.collect()
    return data, output_fields


def _rules_for_context(context: ConversionContext) -> tuple[DerivationRule, ...]:
    if context.sample_time is None:
        return tuple(rule for rule in DERIVATION_RULES if not rule.needs_sample_time)
    return DERIVATION_RULES


@functools.lru_cache(maxsize=32)
def plan_derivations(
    available_fields: TrajectoryField,
    required_fields: TrajectoryField,
    context: ConversionContext,
) -> tuple[DerivationRule, ...] | None:
    """Return the lowest-cost derivation plan for reaching the required fields."""
    context = ConversionContext(context.sample_time, context.group_by)

    if (available_fields & required_fields) == required_fields:
        return ()

    rules = _rules_for_context(context)

    @functools.cache
    def solve(state: TrajectoryField) -> tuple[int, tuple[DerivationRule, ...] | None]:
        if (state & required_fields) == required_fields:
            return 0, ()

        best_cost = 2**31 - 1
        best_plan: tuple[DerivationRule, ...] | None = None

        for rule in rules:
            if (state & rule.requires) != rule.requires:
                continue
            if (state & rule.outputs) == rule.outputs:
                continue

            next_state = state | rule.outputs
            tail_cost, tail_plan = solve(next_state)
            if tail_plan is None:
                continue

            total_cost = rule.cost + tail_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_plan = (rule, *tail_plan)

        return int(best_cost), best_plan

    _, plan = solve(available_fields)
    return plan


def _require_sample_time(sample_time: float | None) -> float:
    if sample_time is None:
        msg = "Scene schema conversion requires sample_time to derive kinematics."
        raise TrajectorySchemaError(msg)
    return sample_time


def _apply_derivative(
    data: pl.LazyFrame,
    context: ConversionContext,
    *,
    x_col: str,
    y_col: str,
    order: int,
    rename: dict[int, list[str]],
    include_intermediate: bool = False,
    sample_time: float | None = None,
) -> pl.LazyFrame:
    """Shared helper for derivative-based field derivation."""
    return derivative(
        data,
        x_col,
        y_col,
        sample_time=_require_sample_time(context.sample_time)
        if sample_time is None
        else sample_time,
        order=order,
        include_intermediate=include_intermediate,
        group_by=context.group_by,
        derivative_rename=rename,
    )


def _velocity_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(data, context, x_col="x", y_col="y", order=1, rename={1: ["vx", "vy"]})


def _acceleration_from_velocity(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="vx",
        y_col="vy",
        order=1,
        rename={1: ["ax", "ay"]},
    )


def _acceleration_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(data, context, x_col="x", y_col="y", order=2, rename={2: ["ax", "ay"]})


def _kinematics_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="x",
        y_col="y",
        order=2,
        rename=_DERIVATIVE_RENAME,
        include_intermediate=True,
    )


def _yaw_from_velocity(data: pl.LazyFrame, _context: ConversionContext) -> pl.LazyFrame:
    return yaw_from_velocity(data, "vx", "vy", "yaw")


def _yaw_from_position(data: pl.LazyFrame, _context: ConversionContext) -> pl.LazyFrame:
    return yaw_from_position(data, "x", "y", "yaw")


DERIVATION_RULES: Final[tuple[DerivationRule, ...]] = (
    DerivationRule(
        name="velocity_from_position",
        requires=_POSITION_FIELDS,
        outputs=_VELOCITY_FIELDS,
        cost=10,
        apply=_velocity_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="acceleration_from_velocity",
        requires=_VELOCITY_FIELDS,
        outputs=_ACCELERATION_FIELDS,
        cost=10,
        apply=_acceleration_from_velocity,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="acceleration_from_position",
        requires=_POSITION_FIELDS,
        outputs=_ACCELERATION_FIELDS,
        cost=13,
        apply=_acceleration_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="kinematics_from_position",
        requires=_POSITION_FIELDS,
        outputs=_KINEMATIC_FIELDS,
        cost=14,
        apply=_kinematics_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="yaw_from_velocity",
        requires=_VELOCITY_FIELDS,
        outputs=_YAW_FIELDS,
        cost=5,
        apply=_yaw_from_velocity,
    ),
    DerivationRule(
        name="yaw_from_position",
        requires=_POSITION_FIELDS,
        outputs=_YAW_FIELDS,
        cost=6,
        apply=_yaw_from_position,
    ),
)
