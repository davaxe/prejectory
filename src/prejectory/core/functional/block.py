# ty: ignore[invalid-assignment, invalid-return-type]
# Work around lost constrained DataFrameT relationships in ty.
# https://github.com/astral-sh/ty/issues/688

"""Block-oriented transforms used for temporal split assignment."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def cumulative_blocks[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    weights: Sequence[float],
    *,
    time_column: str = "frame",
    group_by: Sequence[str] | None = None,
    gap: int = 0,
    remove_gap: bool = True,
    offset_time_column: bool = True,
    partition_column: str = "block",
) -> DataFrameT:
    """Assign contiguous temporal partitions to a DataFrame.

    The parameters `remove_gap` and `offset_time_column` are by default set to
    `True` (as this is the common use case) but can be disabled if the
    original time column and/or the rows in the gap are needed for downstream
    tasks.

    Parameters
    ----------
    data : DataFrameT
        Input DataFrame to partition.
    weights : Sequence[float]
        Relative weights for each partition. The number of partitions is
        determined by the length of this sequence.
    time_column : str, optional
        Name of the time column used for temporal partitioning.
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently.
    gap : int, optional
        Number of excluded timesteps inserted between adjacent partitions.
    remove_gap : bool, optional
        Whether to remove rows that fall into the excluded gap. Default is
        True.
    offset_time_column : bool, optional
        Whether to offset the time column to start from zero inside each
        partition.
        Default is True.
    partition_column : str, optional
        Name of the output partition-id column.

    Returns
    -------
    DataFrameT
        DataFrame with an additional partition-id column indicating the assigned
        partition for each row. If `offset_time_column` is True, the time
        column is offset to start from zero inside each partition.

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "frame": [0, 1, 2, 3, 4, 5, 6, 7],
    ...     "agent_id": [1, 1, 1, 1, 2, 2, 2, 2],
    ...     "value": [11, 12, 13, 14, 20, 21, 22, 23],
    ... })
    >>> cumulative_blocks(df, weights=[0.5, 0.5], time_column="frame", gap=2)
    shape: (6, 4)
    ┌───────┬──────────┬───────┬───────┐
    │ frame ┆ agent_id ┆ value ┆ block │
    │ ---   ┆ ---      ┆ ---   ┆ ---   │
    │ i64   ┆ i64      ┆ i64   ┆ i64   │
    ╞═══════╪══════════╪═══════╪═══════╡
    │ 0     ┆ 1        ┆ 11    ┆ 0     │
    │ 1     ┆ 1        ┆ 12    ┆ 0     │
    │ 2     ┆ 1        ┆ 13    ┆ 0     │
    │ 0     ┆ 2        ┆ 21    ┆ 1     │
    │ 1     ┆ 2        ┆ 22    ┆ 1     │
    │ 2     ┆ 2        ┆ 23    ┆ 1     │
    └───────┴──────────┴───────┴───────┘
    """
    groups = list(group_by) if group_by is not None else []
    weight_sum: float = sum(weights)
    weights_n: list[float] = [w / weight_sum for w in weights]

    min_t: pl.Expr = pl.col(time_column).min()
    max_t: pl.Expr = pl.col(time_column).max()
    if groups:
        min_t = min_t.over(groups)
        max_t = max_t.over(groups)

    n: pl.Expr = max_t - min_t + 1
    n_gaps = max(0, len(weights) - 1)
    total_gap = n_gaps * gap
    usable = n - total_gap
    if isinstance(data, pl.DataFrame):
        usable_min = int(data.select(usable.min()).item())
        if usable_min <= 0:
            msg = (
                "The requested block split leaves no usable frames. "
                "Reduce the gap or number of blocks."
            )
            raise ValueError(msg)
    s: float = 0.0
    cuts: list[pl.Expr] = [pl.lit(0.0)]
    for w in weights_n:
        s += w
        cuts.append((s * usable).round())
    cuts[-1] = usable

    starts = [a + i * gap for i, a in enumerate(cuts[:-1])]
    ends = [b + i * gap for i, b in enumerate(cuts[1:])]

    offset = pl.col(time_column) - min_t
    in_partition = pl.any_horizontal(*[
        ((offset >= start) & (offset < end)) for start, end in zip(starts, ends, strict=True)
    ])

    partition = pl.sum_horizontal(*[(offset >= end).cast(pl.Int64) for end in ends[:-1]])
    data = data.with_columns(
        pl.when(in_partition).then(partition).otherwise(None).alias(partition_column),
    )

    if offset_time_column:
        start_expr = pl.coalesce(*[
            pl.when(pl.col(partition_column) == i).then(start) for i, start in enumerate(starts)
        ])
        data = data.with_columns(
            (pl.col(time_column) - min_t - start_expr).cast(pl.Int64).alias(time_column),
        )

    if remove_gap and gap > 0:
        data = data.filter(pl.col(partition_column).is_not_null())

    return data


def shuffled_blocks[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    weights: Sequence[float],
    n_segments: int,
    *,
    time_column: str = "frame",
    group_by: Sequence[str] | None = None,
    gap: int = 0,
    seed: int | None = None,
    assignment_column: str = "block",
    segment_column: str = "unit",
    offset_time_column: bool = True,
) -> DataFrameT:
    """Assign shuffled split labels to contiguous temporal segments.

    This first creates equally sized contiguous segments, optionally separated
    by gaps, and then deterministically routes each segment to a weighted split
    assignment.

    Parameters
    ----------
    data : DataFrameT
        Input DataFrame to partition and assign.
    weights : Sequence[float]
        Relative weights for each output assignment.
    n_segments : int
        Number of contiguous temporal segments created before shuffling them
        into split assignments.
    time_column : str, optional
        Name of the time column used for temporal partitioning.
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently.
    gap : int, optional
        Number of excluded timesteps inserted between adjacent segments.
    seed : int or None, optional
        Random seed used for the deterministic weighted assignment.
    assignment_column : str, optional
        Name of the output assignment column.
    segment_column : str, optional
        Name of the intermediate contiguous segment-id column.
    offset_time_column : bool, optional
        Whether to offset the time column to start from zero inside each
        segment.

    Returns
    -------
    DataFrameT
        DataFrame with both the contiguous segment id and the shuffled
        assignment column. Split assignments are allocated from a deterministic
        shuffle of the segment ids using rounded cumulative quotas, so each
        split receives a segment count that tracks the requested weights as
        closely as possible.
    """
    data_segments = cumulative_blocks(
        data,
        weights=[1.0] * n_segments,
        time_column=time_column,
        group_by=group_by,
        gap=gap,
        partition_column=segment_column,
        offset_time_column=offset_time_column,
    )
    return _assign_weighted_groups(
        data_segments,
        group_column=segment_column,
        weights=weights,
        group_by=group_by,
        out_col=assignment_column,
        seed=seed,
    )


def _assign_weighted_groups[DataFrameT: (pl.DataFrame, pl.LazyFrame)](
    data: DataFrameT,
    group_column: str,
    weights: Sequence[float],
    *,
    group_by: Sequence[str] | None = None,
    out_col: str = "new_group",
    seed: int | None = None,
) -> DataFrameT:
    groups_by = list(group_by) if group_by is not None else []
    hash_seed = seed if seed is not None else random.randint(0, 2**64 - 1)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    key_columns = [*groups_by, group_column]
    assignment_column = out_col if out_col != group_column else "_assigned_group"
    unique_groups = data.select(*key_columns).unique()
    if isinstance(unique_groups, pl.LazyFrame):
        # We have to know the number of unique groups to allocate the counts, so
        # we need to collect here.
        unique_groups = unique_groups.collect()

    shuffled_groups = unique_groups.with_columns(
        pl
        .struct(*[pl.col(name) for name in key_columns])
        .hash(seed=hash_seed)
        .alias("_shuffle_key"),
    )
    shuffled_groups = shuffled_groups.sort(
        [*groups_by, "_shuffle_key"] if groups_by else "_shuffle_key",
    )
    assignment_frames: list[pl.DataFrame] = []
    grouped_frames = (
        shuffled_groups.partition_by(groups_by, maintain_order=True, as_dict=False)
        if groups_by
        else [shuffled_groups]
    )
    for frame in grouped_frames:
        counts = _allocate_group_counts(frame.height, weights)
        values = [group for group, count in enumerate(counts) for _ in range(count)]
        assignment_frames.append(
            frame.with_columns(pl.Series(name=assignment_column, values=values)),
        )

    assignments = pl.concat(assignment_frames).select(*key_columns, assignment_column)
    if isinstance(data, pl.LazyFrame):
        result = data.join(assignments.lazy(), on=key_columns, how="left")
    else:
        result = data.join(assignments, on=key_columns, how="left")

    if assignment_column != out_col:
        result = result.with_columns(pl.col(assignment_column).alias(out_col)).drop(
            assignment_column,
        )
    return result


def _allocate_group_counts(total_items: int, weights: Sequence[float]) -> list[int]:
    if total_items <= 0:
        return [0] * len(weights)

    total_weight = float(sum(weights))
    normalized_weights = [weight / total_weight for weight in weights]
    exact_counts = [weight * total_items for weight in normalized_weights]
    counts = [int(count) for count in exact_counts]
    remaining = total_items - sum(counts)

    fractional_order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in fractional_order[:remaining]:
        counts[index] += 1

    return counts
