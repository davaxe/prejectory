"""Map-graph builder for the highD dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import EdgeType
from prejectory.processing.maps import (
    FeatureMapBuilder,
    MapProvider,
    MapReference,
    PathFeature,
    extract_configured_map,
    resolve_map_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.config.models import MapConfig
    from prejectory.core.maps import MapGraph
    from prejectory.core.scene import Scene


@dataclass(frozen=True, slots=True)
class HighDMapProvider(MapProvider):
    config: MapConfig
    margin_fraction: float = 0.1

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = resolve_map_key(scene, reference)
        if key is None:
            return None

        min_x = scene.frame.select(pl.col("x")).min().item()
        max_x = scene.frame.select(pl.col("x")).max().item()
        span = max_x - min_x

        builder = HighDMapBuilder(
            Path(str(key)),
            min_x - span * self.margin_fraction,
            max_x + span * self.margin_fraction,
        )
        map_graph = builder.build(
            min_distance=self.config.min_distance,
            interpolation_distance=self.config.interpolation_distance,
        )
        return extract_configured_map(map_graph, scene, self.config)


class HighDMapBuilder(FeatureMapBuilder):
    """Map builder for the highD dataset."""

    def __init__(self, meta_file: Path, start_x: float, end_x: float) -> None:
        self._start_x: float = start_x
        self._end_x: float = end_x
        self._meta_file: Path = meta_file

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        data = pl.read_csv(self._meta_file).select(
            pl.col("upperLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
            pl.col("lowerLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
        )

        n_lane_markings = len(data["upperLaneMarkings"][0])
        for i, y in enumerate(data["upperLaneMarkings"][0]):
            yield PathFeature(
                points=((self._start_x, y), (self._end_x, y)),
                edge_types=(
                    EdgeType.ROAD_BORDER
                    if i == 0 or i == n_lane_markings - 1
                    else EdgeType.LINE_THIN_DASHED
                ),
            )

        n_lane_markings = len(data["lowerLaneMarkings"][0])
        for i, y in enumerate(data["lowerLaneMarkings"][0]):
            yield PathFeature(
                points=((self._start_x, y), (self._end_x, y)),
                edge_types=(
                    EdgeType.ROAD_BORDER
                    if i == 0 or i == n_lane_markings - 1
                    else EdgeType.LINE_THIN_DASHED
                ),
            )
