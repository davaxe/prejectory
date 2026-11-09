# ty: ignore[invalid-assignment, invalid-return-type, invalid-argument-type]
# Work around lost constrained DataFrameT relationships in ty.
# https://github.com/astral-sh/ty/issues/688
# https://github.com/astral-sh/ty/issues/3605

"""Sliding-window transforms for turning trajectories into scene windows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from prejectory.core.functional.basic import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence


WindowPolicy = Literal["strict", "anchored", "partial"]

_WINDOW_START_COLUMN = "_prejectory_window_start"
_WINDOW_FIRST_COLUMN = "_prejectory_window_first"
_WINDOW_GROUP_MAX_COLUMN = "_prejectory_window_group_max"
_WINDOW_ROW_ORDER_COLUMN = "_prejectory_window_row_order"
_WINDOW_FRAMES_COLUMN = "_prejectory_window_frames"


def sliding_window[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    policy: WindowPolicy = "strict",
    group_by: str | Sequence[str] | None = None,
    is_sorted: bool = False,
    offset_sliding_col: bool = False,
) -> DataFrameT:
    """Generate sliding windows from a DataFrame.

    Parameters
    ----------
    data : DataFrameT
        Input DataFrame to generate windows from.
    window_size : int
        Temporal span of each window in units of `sliding_col`.
    step_size : int
        Distance between consecutive window starts in units of `sliding_col`.
    sliding_col : str, optional
        Column name to use for determining the window boundaries.
        Defaults to "frame".
    group_by : str, optional
        Column name(s) to group by before applying the sliding window.
        This allows for generating windows within each group separately.
    is_sorted : bool, optional
        Whether the input DataFrame is already sorted by `sliding_col`.
        If False, the DataFrame will be sorted before generating windows.
        Defaults to False.

    Returns
    -------
    DataFrameT
        Adds a `window_index` column to the input DataFrame indicating the window
        each row belongs to.
    """
    group_keys = list(normalize_group_by(group_by))

    if not is_sorted:
        data = data.sort([*group_keys, sliding_col] if group_keys else sliding_col)

    windows = _create_windows(
        data,
        window_size,
        step_size,
        sliding_col=sliding_col,
        policy=policy,
        group_by=group_keys,
    )
    return _explode_windows(windows, sliding_col, group_keys, offset_sliding_col=offset_sliding_col)


def _create_windows[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    window_size: int,
    step_size: int,
    *,
    sliding_col: str,
    policy: WindowPolicy,
    group_by: list[str],
) -> DataFrameT:
    window_index = _create_window_index(
        data,
        window_size,
        step_size,
        sliding_col=sliding_col,
        policy=policy,
        group_by=group_by,
    )

    lower_bound = pl.max_horizontal(pl.lit(0), pl.col(sliding_col) - window_size + 1)
    first_window_start = ((lower_bound + step_size - 1) // step_size) * step_size
    last_window_start = (pl.col(sliding_col) // step_size) * step_size
    window_columns = [*group_by, _WINDOW_START_COLUMN]

    expanded = (
        data
        .with_row_index(_WINDOW_ROW_ORDER_COLUMN)
        .with_columns(
            pl.int_ranges(first_window_start, last_window_start + step_size, step_size).alias(
                _WINDOW_START_COLUMN,
            ),
        )
        .explode(_WINDOW_START_COLUMN, empty_as_null=True)
    )
    return expanded.join(window_index, on=window_columns, how="inner")


def _create_window_index[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    window_size: int,
    step_size: int,
    *,
    sliding_col: str,
    policy: WindowPolicy,
    group_by: list[str],
) -> DataFrameT:
    if policy == "anchored":
        max_expr = pl.col(sliding_col).max()
        if group_by:
            max_expr = max_expr.over(group_by)
        data = data.with_columns(max_expr.alias(_WINDOW_GROUP_MAX_COLUMN))

    aggs = [pl.col(sliding_col).alias(_WINDOW_FRAMES_COLUMN)]
    if policy == "anchored":
        aggs.append(pl.col(_WINDOW_GROUP_MAX_COLUMN).alias(_WINDOW_GROUP_MAX_COLUMN))

    grouped = data.group_by_dynamic(
        sliding_col,
        every=f"{step_size}i",
        period=f"{window_size}i",
        group_by=group_by or None,
    ).agg(*aggs)

    if policy == "strict":
        span = (
            pl.col(_WINDOW_FRAMES_COLUMN).list.last()
            - pl.col(_WINDOW_FRAMES_COLUMN).list.first()
            + 1
        )
        grouped = grouped.filter(span == window_size)
    elif policy == "anchored":
        grouped = grouped.filter(
            (pl.col(sliding_col) + window_size - 1)
            <= pl.col(_WINDOW_GROUP_MAX_COLUMN).list.first(),
        )

    return (
        grouped
        .with_columns(pl.col(_WINDOW_FRAMES_COLUMN).list.first().alias(_WINDOW_FIRST_COLUMN))
        .rename({sliding_col: _WINDOW_START_COLUMN})
        .with_row_index("window_index")
        .select(*group_by, _WINDOW_START_COLUMN, "window_index", _WINDOW_FIRST_COLUMN)
    )


def _explode_windows[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    windows: DataFrameT,
    sliding_col: str,
    group_keys: list[str],
    *,
    offset_sliding_col: bool,
    extra_exclude_cols: Sequence[str] = (),
) -> DataFrameT:
    """Finalize expanded rows into scene-window rows."""
    exclude_cols = [
        _WINDOW_START_COLUMN,
        _WINDOW_FIRST_COLUMN,
        _WINDOW_GROUP_MAX_COLUMN,
        _WINDOW_ROW_ORDER_COLUMN,
        *extra_exclude_cols,
    ]
    schema_names = (
        windows.collect_schema().names()
        if isinstance(windows, pl.LazyFrame)
        else windows.schema.names()
    )
    available_exclude_cols = [col for col in exclude_cols if col in schema_names]
    frame_expr = (
        pl.col(sliding_col) - pl.col(_WINDOW_FIRST_COLUMN)
        if offset_sliding_col
        else pl.col(sliding_col)
    )
    result = (
        windows
        .with_columns(frame_expr.alias(sliding_col))
        .sort("window_index", _WINDOW_ROW_ORDER_COLUMN)
        .drop(available_exclude_cols)
    )
    result_names = (
        result.collect_schema().names()
        if isinstance(result, pl.LazyFrame)
        else result.schema.names()
    )
    preferred_order = ["window_index", *group_keys, sliding_col]
    ordered_names = [
        *[name for name in preferred_order if name in result_names],
        *[name for name in result_names if name not in preferred_order],
    ]
    return result.select(ordered_names)
