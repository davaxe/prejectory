"""Temporal resampling utilities for trajectory tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

from prejectory.core.errors import LoaderConfigError
from prejectory.core.functional.basic import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prejectory.core.typing import DataFrameT

SEGMENT_COLUMN: Final = "_resample_segment"

EmittedDerivative = Literal["velocity", "acceleration"]
Interpolator = PPoly
InterpolatorFactory = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], Interpolator]


class ResampleMethod(StrEnum):
    """Interpolation strategy used during resampling."""

    LINEAR = "linear"
    CUBIC = "cubic"
    PCHIP = "pchip"


def _velocity_column_name(coordinate: str) -> str:
    return f"v{coordinate}" if len(coordinate) == 1 else f"velocity_{coordinate}"


def _acceleration_column_name(coordinate: str) -> str:
    return f"a{coordinate}" if len(coordinate) == 1 else f"acceleration_{coordinate}"


@dataclass(frozen=True, slots=True)
class ResampleSpec:
    """Validated specification for temporal resampling."""

    up: int = 1
    down: int = 1
    method: ResampleMethod = ResampleMethod.LINEAR
    coordinates: tuple[str, ...] = ("x", "y")
    emit_velocity: bool = False
    emit_acceleration: bool = False
    max_gap: int = 1
    sample_time: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize the resampling factors."""
        if self.up <= 0 or self.down <= 0:
            msg = "Resampling factors 'up' and 'down' must be positive."
            raise LoaderConfigError(msg)
        if self.max_gap <= 0:
            msg = "Resampling 'max_gap' must be positive."
            raise LoaderConfigError(msg)
        if self.sample_time <= 0.0:
            msg = "Resampling 'sample_time' must be positive."
            raise LoaderConfigError(msg)

        simplified_up, simplified_down = Fraction(self.up, self.down).as_integer_ratio()
        object.__setattr__(self, "up", simplified_up)
        object.__setattr__(self, "down", simplified_down)

        if self.method is ResampleMethod.LINEAR and (self.emit_acceleration or self.emit_velocity):
            msg = "Linear resampling does not support emitting derivatives."
            raise LoaderConfigError(msg)

    @classmethod
    def cubic(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for cubic spline resampling."""
        return cls(up=up, down=down, method=ResampleMethod.CUBIC)

    @classmethod
    def pchip(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for PCHIP resampling."""
        return cls(up=up, down=down, method=ResampleMethod.PCHIP)

    @classmethod
    def linear(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for linear resampling."""
        return cls(up=up, down=down, method=ResampleMethod.LINEAR)

    def _with_update(self, **kwargs: object) -> ResampleSpec:
        return replace(self, **kwargs)

    def with_sample_time(self, sample_time: float) -> ResampleSpec:
        """Return a copy with a different sampling interval."""
        return self._with_update(sample_time=sample_time)

    def with_emit(
        self,
        *,
        acceleration: bool | None = None,
        velocity: bool | None = None,
    ) -> ResampleSpec:
        """Return a copy with different derivative emission settings."""
        emit_velocity = self.emit_velocity if velocity is None else velocity
        emit_acceleration = self.emit_acceleration if acceleration is None else acceleration
        return self._with_update(emit_velocity=emit_velocity, emit_acceleration=emit_acceleration)

    @property
    def no_resampling(self) -> bool:
        """Whether the specification keeps the original sampling rate."""
        return self.up == 1 and self.down == 1


@dataclass(frozen=True, slots=True)
class ResamplePlan:
    """Execution-ready resampling plan derived from a :class:`ResampleSpec`."""

    frame_column: str
    group_by: tuple[str, ...]
    time_origin_by: tuple[str, ...]
    segment_keys: tuple[str, ...]
    coordinates: tuple[str, ...]
    velocity_columns: tuple[str, ...]
    acceleration_columns: tuple[str, ...]
    emitted_columns: tuple[str, ...]

    @property
    def emit_velocity(self) -> bool:
        """Whether the plan emits velocity columns."""
        return len(self.velocity_columns) > 0

    @property
    def emit_acceleration(self) -> bool:
        """Whether the plan emits acceleration columns."""
        return len(self.acceleration_columns) > 0

    @property
    def packed_columns(self) -> tuple[str, ...]:
        """Frame and coordinate columns packed before spline interpolation."""
        return (self.frame_column, *self.coordinates)


def resample(
    data: DataFrameT,
    spec: ResampleSpec | None = None,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    time_origin_by: str | Sequence[str] | None = None,
) -> DataFrameT:
    """Resample trajectory data according to an explicit resampling spec.

    Parameters
    ----------
    data : DataFrameT
        Polars `DataFrame` or `LazyFrame` containing the trajectory data to
        resample.
    spec : ResampleSpec | None, optional
        Resampling specification. When omitted, the default
        :class:`ResampleSpec` is used.
    frame_column : str, optional
        Column containing monotonically increasing frame indices within each
        trajectory group.
    group_by : str or sequence of str or None, optional
        Column or columns that define independent trajectories. When `None`,
        the full table is treated as one trajectory.
    time_origin_by : str or sequence of str or None, optional
        Column or columns that define a shared temporal origin. Frame indices
        are normalized relative to the minimum frame in each origin group before
        resampling. When omitted, `group_by` is used.

    Returns
    -------
    DataFrameT
        Resampled table of the same eager/lazy type as `data`.
    """
    resample_spec = spec or ResampleSpec()
    plan = _resolve_request(
        resample_spec,
        frame_column=frame_column,
        group_by=group_by,
        time_origin_by=time_origin_by,
    )
    match resample_spec.method:
        case ResampleMethod.LINEAR:
            return _linear_resample(data=data, spec=resample_spec, plan=plan)
        case ResampleMethod.CUBIC:
            return _spline_resample(
                data=data,
                spec=resample_spec,
                plan=plan,
                interpolator_factory=_cubic_spline_interpolator_factory,
            )
        case ResampleMethod.PCHIP:
            return _spline_resample(
                data=data,
                spec=resample_spec,
                plan=plan,
                interpolator_factory=_pchip_interpolator_factory,
            )


def _resolve_request(
    spec: ResampleSpec,
    *,
    frame_column: str,
    group_by: str | Sequence[str] | None,
    time_origin_by: str | Sequence[str] | None,
) -> ResamplePlan:
    """Normalize a resampling specification into an execution plan."""
    group_columns = normalize_group_by(group_by)
    time_origin_columns = (
        group_columns if time_origin_by is None else normalize_group_by(time_origin_by)
    )
    velocity_columns = (
        tuple(_velocity_column_name(coordinate) for coordinate in spec.coordinates)
        if spec.emit_velocity
        else ()
    )
    acceleration_columns = (
        tuple(_acceleration_column_name(coordinate) for coordinate in spec.coordinates)
        if spec.emit_acceleration
        else ()
    )
    emitted_columns = (frame_column, *spec.coordinates, *velocity_columns, *acceleration_columns)
    return ResamplePlan(
        frame_column=frame_column,
        group_by=group_columns,
        time_origin_by=time_origin_columns,
        segment_keys=(*group_columns, SEGMENT_COLUMN),
        coordinates=spec.coordinates,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        emitted_columns=emitted_columns,
    )


def _segment_data(
    data: DataFrameT,
    *,
    frame_column: str,
    group_by: Sequence[str],
    max_gap: int,
) -> DataFrameT:
    """Annotate contiguous trajectory segments separated by gaps larger than `max_gap`."""
    data = data.sort([*group_by, frame_column])
    expr = (pl.col(frame_column).diff() > max_gap).fill_null(value=False).cum_sum()
    if group_by:
        expr = expr.over(group_by)

    return data.with_columns(expr.alias(SEGMENT_COLUMN))


def _normalize_frame_origin(data: DataFrameT, plan: ResamplePlan) -> DataFrameT:
    """Normalize frame indices relative to the configured temporal origin."""
    origin = pl.col(plan.frame_column).min()
    if plan.time_origin_by:
        origin = origin.over(plan.time_origin_by)
    return data.with_columns((pl.col(plan.frame_column) - origin).alias(plan.frame_column))


def _packed_struct_expression(plan: ResamplePlan) -> pl.Expr:
    """Return a struct expression containing the columns resampling operates on."""
    return pl.struct(pl.col(column) for column in plan.packed_columns)


def _not_packed_columns(plan: ResamplePlan) -> pl.Expr:
    """Return an expression selecting columns carried through unchanged."""
    excluded = [*plan.group_by, *plan.emitted_columns, SEGMENT_COLUMN]
    return pl.all().exclude(excluded)


def _linear_resample(data: DataFrameT, spec: ResampleSpec, plan: ResamplePlan) -> DataFrameT:
    """Resample trajectory data using linear interpolation and carry-forward values."""
    if spec.no_resampling:
        return data
    data = _normalize_frame_origin(data, plan)
    data = _segment_data(
        data,
        frame_column=plan.frame_column,
        group_by=plan.group_by,
        max_gap=spec.max_gap,
    )

    if spec.up > 1:
        data = _upsample_dataframe(data, factor=spec.up, plan=plan)
    if spec.down > 1:
        data = _downsample_dataframe(data, factor=spec.down, frame_column=plan.frame_column)

    return data.drop(SEGMENT_COLUMN, strict=False)


def _downsample_dataframe(data: DataFrameT, *, factor: int, frame_column: str) -> DataFrameT:
    return data.filter(pl.col(frame_column) % factor == 0).with_columns(
        (pl.col(frame_column) // factor).alias(frame_column),
    )


def _upsample_dataframe(data: DataFrameT, *, factor: int, plan: ResamplePlan) -> DataFrameT:
    scaled = data.with_columns((pl.col(plan.frame_column) * factor).alias(plan.frame_column))
    frame_range = pl.int_range(
        pl.col(plan.frame_column).min(),
        pl.col(plan.frame_column).max() + 1,
        step=1,
        dtype=pl.Int64,
    ).alias(plan.frame_column)

    if plan.segment_keys:
        upsampled = (
            scaled
            .group_by(plan.segment_keys, maintain_order=True)
            .agg(frame_range)
            .explode(plan.frame_column)
        )
    else:
        upsampled = scaled.select(frame_range)

    on = [*plan.segment_keys, plan.frame_column]
    exprs: list[pl.Expr] = []

    if plan.coordinates:
        interpolated = cs.by_name(*plan.coordinates, require_all=False).interpolate()
        exprs.append(interpolated.over(plan.segment_keys) if plan.segment_keys else interpolated)

    carried = pl.exclude(*on, *plan.coordinates).forward_fill()
    exprs.append(carried.over(plan.segment_keys) if plan.segment_keys else carried)
    return (
        upsampled.join(cast("DataFrameT", scaled), on=on, how="left").sort(on).with_columns(exprs)
    )


def _spline_resample(
    data: DataFrameT,
    spec: ResampleSpec,
    plan: ResamplePlan,
    *,
    interpolator_factory: InterpolatorFactory,
) -> DataFrameT:
    """Resample trajectory data with a SciPy spline interpolator."""
    if spec.no_resampling and not plan.emit_velocity and not plan.emit_acceleration:
        return data.sort([*plan.group_by, plan.frame_column]).drop(SEGMENT_COLUMN, strict=False)

    data = _normalize_frame_origin(data, plan)
    segmented = _segment_data(
        data,
        frame_column=plan.frame_column,
        group_by=plan.group_by,
        max_gap=spec.max_gap,
    )

    return (
        segmented
        .group_by(plan.segment_keys, maintain_order=True)
        .agg(
            _packed_struct_expression(plan)
            .map_batches(
                lambda series: _resample_segment(
                    series,
                    plan=plan,
                    up=spec.up,
                    down=spec.down,
                    sample_time=spec.sample_time,
                    interpolator_factory=interpolator_factory,
                ),
                return_dtype=pl.Struct(_output_dtype(plan)),
            )
            .alias("_resampled"),
            _not_packed_columns(plan).first(),
        )
        .explode("_resampled")
        .unnest("_resampled")
        .drop(SEGMENT_COLUMN, strict=False)
    )


def _output_dtype(plan: ResamplePlan) -> dict[str, type[pl.DataType]]:
    dtype: dict[str, type[pl.DataType]] = {plan.frame_column: pl.Int64}
    for column in (*plan.coordinates, *plan.velocity_columns, *plan.acceleration_columns):
        dtype[column] = pl.Float64
    return dtype


def _cubic_spline_interpolator_factory(
    time_old: npt.NDArray[np.float64],
    coordinates_old: npt.NDArray[np.float64],
) -> Interpolator:
    """Build a natural cubic-spline interpolator for one trajectory segment."""
    return CubicSpline(time_old, coordinates_old, axis=0, bc_type="natural")


def _pchip_interpolator_factory(
    time_old: npt.NDArray[np.float64],
    coordinates_old: npt.NDArray[np.float64],
) -> Interpolator:
    """Build a monotone PCHIP interpolator for one trajectory segment."""
    return PchipInterpolator(time_old, coordinates_old, axis=0)


def _resample_segment(
    series: pl.Series,
    *,
    plan: ResamplePlan,
    up: int,
    down: int,
    sample_time: float,
    interpolator_factory: InterpolatorFactory,
) -> pl.Series:
    df = series.struct.unnest()

    if df.height <= 1:
        return _resample_single_point(series.name, df, plan, up=up, down=down)

    return _resample_multi_point(
        series.name,
        df,
        plan,
        up=up,
        down=down,
        sample_time=sample_time,
        interpolator_factory=interpolator_factory,
    )


def _resample_single_point(
    name: str,
    df: pl.DataFrame,
    plan: ResamplePlan,
    *,
    up: int,
    down: int,
) -> pl.Series:
    frames_old = df[plan.frame_column].to_numpy().astype(np.float64, copy=False)
    coordinates_old = df.select(tuple(plan.coordinates)).to_numpy().astype(np.float64, copy=False)
    out: dict[str, object] = {plan.frame_column: (frames_old * (up / down)).astype(np.int64)}

    for index, column in enumerate(plan.coordinates):
        out[column] = coordinates_old[:, index]
    for columns in (plan.velocity_columns, plan.acceleration_columns):
        for column in columns:
            out[column] = np.full(df.height, np.nan, dtype=np.float64)

    return pl.DataFrame(out).select(plan.emitted_columns).to_struct(name)


def _resample_multi_point(
    name: str,
    df: pl.DataFrame,
    plan: ResamplePlan,
    *,
    up: int,
    down: int,
    sample_time: float,
    interpolator_factory: InterpolatorFactory,
) -> pl.Series:
    frame_old = df[plan.frame_column].to_numpy().astype(np.float64, copy=False)
    time_old = frame_old * sample_time
    coordinates_old = df.select(tuple(plan.coordinates)).to_numpy().astype(np.float64, copy=False)

    step_frames = down / up
    step_time = step_frames * sample_time
    time_start = float(time_old[0])
    time_end = float(time_old[-1])
    n_new = int(np.floor((time_end - time_start) / step_time)) + 1
    time_new = time_start + np.arange(n_new, dtype=np.float64) * step_time
    interpolator = interpolator_factory(time_old, coordinates_old)

    out: dict[str, object] = {
        plan.frame_column: np.arange(n_new, dtype=np.int64) + int(frame_old[0] * up / down),
    }
    coordinates_new = np.asarray(interpolator(time_new), dtype=np.float64)
    for index, column in enumerate(plan.coordinates):
        out[column] = coordinates_new[:, index]

    if plan.emit_velocity:
        velocity = np.asarray(interpolator(time_new, nu=1), dtype=np.float64)
        for index, column in enumerate(plan.velocity_columns):
            out[column] = velocity[:, index]

    if plan.emit_acceleration:
        acceleration = np.asarray(interpolator(time_new, nu=2), dtype=np.float64)
        for index, column in enumerate(plan.acceleration_columns):
            out[column] = acceleration[:, index]

    return pl.DataFrame(out).select(plan.emitted_columns).to_struct(name)
