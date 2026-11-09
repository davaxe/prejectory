from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import numpy as np
import polars as pl
from scipy.interpolate import UnivariateSpline

from prejectory.core.categories import EdgeType
from prejectory.processing.maps import FeatureMapBuilder, PathFeature, Point

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

LaneId = int | str


@dataclass(slots=True, frozen=True)
class LaneDescription:
    """Describes the lanes on a highway, including their IDs and directions."""

    ids: list[LaneId]
    direction: list[bool]


class HighwayLaneMapBuilder(FeatureMapBuilder):
    """Construct a lane graph for a highway based on vehicle trajectory data."""

    def __init__(
        self,
        data: pl.LazyFrame,
        id_col: str = "id",
        x_col: str = "x",
        y_col: str = "y",
        lane_id_col: str = "lane_id",
        *,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        bin_size: float = 8.0,
        include_outer_borders: bool = True,
        smoothing: float | None = None,
        lane_description: LaneDescription | None = None,
    ) -> None:
        self._data: pl.LazyFrame = data
        self._id_col: str = id_col
        self._x_col: str = x_col
        self._y_col: str = y_col
        self._lane_id_col: str = lane_id_col
        self._orientation: Literal["vertical", "horizontal"] = orientation
        self._bin_size: float = bin_size
        self._include_outer_borders: bool = include_outer_borders
        self._smoothing_factor: float | None = smoothing
        self._lane_description: LaneDescription | None = lane_description
        if orientation == "vertical":
            self._long_col: str = self._y_col
            self._lat_col: str = self._x_col
        else:
            self._long_col = self._x_col
            self._lat_col = self._y_col

    @classmethod
    def from_csv(
        cls,
        *file: Path,
        select_expr: Iterable[pl.Expr],
        bin_size: float = 8.0,
        include_outer_borders: bool = True,
        smoothing: float | None = None,
        lane_description: LaneDescription | None = None,
    ) -> HighwayLaneMapBuilder:
        """Create a map builder from CSV files containing trajectory data."""
        data = pl.scan_csv(source=list(file)).select(select_expr)
        return cls(
            data,
            bin_size=bin_size,
            include_outer_borders=include_outer_borders,
            smoothing=smoothing,
            lane_description=lane_description,
        )

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        data = self._data
        lane_description = self._lane_description
        if lane_description is not None:
            mapping: dict[LaneId, int] = {
                lane_id: idx for idx, lane_id in enumerate(lane_description.ids)
            }
            data = data.with_columns(
                pl.col(self._lane_id_col).replace_strict(mapping).alias(self._lane_id_col),
            )
            lane_description = LaneDescription(
                ids=list(range(len(lane_description.ids))),
                direction=lane_description.direction,
            )

        lane_centers = self._get_lane_centers(data, bin_size=self._bin_size)
        if self._smoothing_factor is not None:
            lane_centers = _smooth_lines(
                lane_centers,
                group_col=self._lane_id_col,
                x_col="lat_center",
                y_col="long_bin",
                smoothing_factor=self._smoothing_factor,
            ).lazy()

        lane_borders = self._get_inner_borders(lane_centers)
        for (left, right), group_data in lane_borders.collect().group_by([
            "left_lane",
            "right_lane",
        ]):
            if self._orientation == "vertical":
                x_vals, y_vals = group_data["border_lat"], group_data["long_bin"]
            else:
                x_vals, y_vals = group_data["long_bin"], group_data["border_lat"]
            points: list[Point] = list(zip(x_vals.to_list(), y_vals.to_list(), strict=True))
            yield PathFeature(
                points=tuple(points),
                edge_types=self._get_border_type(left, right, lane_description),
            )

        if self._include_outer_borders:
            outer_borders = self._get_outer_borders(lane_centers, lane_borders)
            for _, group_data in outer_borders.collect().group_by("border_type"):
                if self._orientation == "vertical":
                    x_vals, y_vals = group_data["border_lat"], group_data["long_bin"]
                else:
                    x_vals, y_vals = group_data["long_bin"], group_data["border_lat"]
                points = list(zip(x_vals.to_list(), y_vals.to_list(), strict=True))
                yield PathFeature(points=tuple(points), edge_types=EdgeType.CURB)

    def _get_lane_centers(self, data: pl.LazyFrame, bin_size: float = 8.0) -> pl.LazyFrame:
        return (
            data
            .with_columns((pl.col(self._long_col) // bin_size * bin_size).alias("long_bin"))
            .group_by([self._lane_id_col, "long_bin"])
            .agg(pl.col(self._lat_col).median().alias("lat_center"))
            .sort([self._lane_id_col, "long_bin"])
        )

    def _get_inner_borders(self, lane_centers: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lane_centers
            .with_columns((pl.col(self._lane_id_col) + 1).alias("next_lane_id"))
            .join(
                lane_centers,
                left_on=["next_lane_id", "long_bin"],
                right_on=[self._lane_id_col, "long_bin"],
                suffix="_right",
            )
            .select([
                "long_bin",
                pl.col(self._lane_id_col).alias("left_lane"),
                pl.col("next_lane_id").alias("right_lane"),
                ((pl.col("lat_center") + pl.col("lat_center_right")) / 2.0).alias("border_lat"),
            ])
        )

    def _get_outer_borders(self, centers: pl.LazyFrame, borders: pl.LazyFrame) -> pl.LazyFrame:
        avg_half_width = (
            borders
            .join(
                centers,
                left_on=["left_lane", "long_bin"],
                right_on=[self._lane_id_col, "long_bin"],
            )
            .select((pl.col("border_lat") - pl.col("lat_center")).abs())
            .collect()
            .to_series()
            .mean()
        )

        extreme_lanes = centers.with_columns([
            pl.col(self._lane_id_col).min().alias("min_id"),
            pl.col(self._lane_id_col).max().alias("max_id"),
        ]).filter(
            (pl.col(self._lane_id_col) == pl.col("min_id"))
            | (pl.col(self._lane_id_col) == pl.col("max_id")),
        )

        return extreme_lanes.select([
            "long_bin",
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.lit("left_outer"))
            .otherwise(pl.lit("right_outer"))
            .alias("border_type"),
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.col("lat_center") - avg_half_width)
            .otherwise(pl.col("lat_center") + avg_half_width)
            .alias("border_lat"),
        ])

    @staticmethod
    def _get_border_type(
        left: LaneId,
        right: LaneId,
        lane_description: LaneDescription | None,
    ) -> EdgeType:
        if lane_description is not None:
            left_idx = lane_description.ids.index(left)
            right_idx = lane_description.ids.index(right)
            if lane_description.direction[left_idx] != lane_description.direction[right_idx]:
                return EdgeType.ROAD_BORDER
            return EdgeType.LINE_THICK_DASHED

        return EdgeType.VIRTUAL


def _smooth_lines(
    df: pl.LazyFrame | pl.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    smoothing_factor: float = 1.0,
) -> pl.DataFrame:
    eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df

    def _apply_spline(group: pl.DataFrame) -> pl.DataFrame:
        unique_group = group.unique(subset=[y_col]).sort(y_col)
        if len(unique_group) < 4:
            return group

        y = unique_group[y_col].to_numpy()
        x = unique_group[x_col].to_numpy()
        spline = UnivariateSpline(y, x, s=smoothing_factor)
        smoothed_x = np.asarray(spline(group[y_col].to_numpy()), dtype=np.float64)
        return group.with_columns(pl.Series(x_col, smoothed_x))

    return eager_df.group_by(group_col).map_groups(_apply_spline)
