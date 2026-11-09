# ty: ignore[invalid-return-type]
# Work around lost constrained DataFrameT relationships in ty.
# https://github.com/astral-sh/ty/issues/688

"""Highway-specific transforms used for lane-change sampling pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from prejectory.core.functional.basic import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence


def valid_lane_change[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    persist: int,
    margin_before: int = 0,
    margin_after: int = 0,
    *,
    frame_column: str = "frame",
    agent_id_column: str = "id",
    lane_id_column: str = "lane_id",
    group_by: str | Sequence[str] | None = None,
    valid_column: str = "valid_lane_change",
) -> DataFrameT:
    """Add a boolean column indicating whether change occurs at each frame.

    This is a wrapper around `valid_lane_change_expr` that handles the necessary
    sorting to ensure correct results. The added column will be named according
    to the `valid_column` parameter (default is "valid_lane_change"). See
    `valid_lane_change_expr` for more details.

    Parameters
    ----------
    data : DataFrameT
        Input DataFrame containing at least the frame, agent id, and lane id
        columns.
    persist : int
        Number of frames the lane change should persist in order to be considered
        valid. This is useful if the lane assignment is noisy.
    margin_before : int, optional
        Number of frames that should exist before the lane change in order
        to be considered valid.
    margin_after : int, optional
        Number of frames that should exist after the lane change in order
        to be considered valid.
    frame_column : str, optional
        Name of the frame column. Default is "frame".
    agent_id_column : str, optional
        Name of the agent id column. Default is "id".
    lane_id_column : str, optional
        Name of the lane id column. Default is "lane_id".
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently. Default is no
        partitioning.
    valid_column : str, optional
        Name of the output boolean column indicating valid lane changes.

    Returns
    -------
    DataFrameT
        Input DataFrame with an additional boolean column indicating valid lane
        changes.

    """
    group_keys = normalize_group_by(group_by)
    sort_columns = [*group_keys, agent_id_column, frame_column]
    return data.sort(sort_columns).with_columns(
        valid_lane_change_expr(
            persist=persist,
            margin_before=margin_before,
            margin_after=margin_after,
            frame_column=frame_column,
            agent_id_column=agent_id_column,
            lane_id_column=lane_id_column,
            group_by=group_keys,
        ).alias(valid_column),
    )


def valid_lane_change_expr(
    persist: int,
    margin_before: int = 0,
    margin_after: int = 0,
    *,
    frame_column: str = "frame",
    agent_id_column: str = "id",
    lane_id_column: str = "lane_id",
    group_by: str | Sequence[str] | None = None,
) -> pl.Expr:
    """Determine all frames where there is a valid lane change.

    A valid lane change must:
        1. Lane assignment changes from the previous frame to the current frame.
        2. The lane change persist (does not go back to same previous lane)
           for at least `persist` frames after the lane change.
        3. Optionally, there must exist at least `margin_before` frames before
           the lane change and at least `margin_after` frames after the lane
           change (regardless of lane assignment)

    Parameters
    ----------
    persist : int
        Number of frames the lane change should persist in order to be considered
        valid. This is useful if the lane assignment is noisy.
    margin_before : int, optional
        Number of frames that should exist before the lane change in order
        to be considered valid.
    margin_after : int, optional
        Number of frames that should exist after the lane change in order
        to be considered valid.
    frame_column : str, optional
        Name of the frame column. Default is "frame".
    agent_id_column : str, optional
        Name of the agent id column. Default is "id".
    lane_id_column : str, optional
        Name of the lane id column. Default is "lane_id".
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently. Default is no
        partitioning.

    Returns
    -------
    pl.Expr
        An expression to calculate boolean mask for rows where a valid lane
        change occurs.

    Notes
    -----
    This function assumes that the input DataFrame is sorted by `group_by`, then
    `agent_id_column`, and then `frame_column`. If this is not the case, the
    results may be incorrect. It is recommended to sort the DataFrame before
    applying this expression, or to use the `valid_lane_change` function which
    handles the sorting internally.


    """
    lane = pl.col(lane_id_column)
    frame = pl.col(frame_column)
    partition_by = [*normalize_group_by(group_by), agent_id_column]

    def lane_at(offset: int) -> pl.Expr:
        return lane.shift(offset).over(partition_by)

    def frame_at(offset: int) -> pl.Expr:
        return frame.shift(offset).over(partition_by)

    prev_lane = lane_at(1)
    prev_frame = frame_at(1)
    changed_now = (prev_frame == frame - 1) & (lane != prev_lane)
    before_conditions = [
        (frame_at(i) == frame - i) & (lane_at(i) == prev_lane) for i in range(1, margin_before + 1)
    ]
    after_conditions = [
        (frame_at(-i) == frame + i) & (lane_at(-i) == lane)
        for i in range(1, persist + margin_after + 1)
    ]

    return pl.all_horizontal([changed_now, *before_conditions, *after_conditions]).fill_null(
        value=False,
    )
