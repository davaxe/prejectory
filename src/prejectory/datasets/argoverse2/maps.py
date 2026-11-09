from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, override

from prejectory.core.categories import EdgeType
from prejectory.processing.maps import (
    FeatureMapBuilder,
    MapProvider,
    MapReference,
    PathFeature,
    Point,
    extract_configured_map,
    resolve_map_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.config.models import MapConfig
    from prejectory.core.maps import MapGraph
    from prejectory.core.scene import Scene


@dataclass(frozen=True, slots=True)
class Argoverse2MapProvider(MapProvider):
    config: MapConfig

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = resolve_map_key(scene, reference)
        if key is None:
            return None
        map_graph = _load_argoverse2_map(
            str(key),
            self.config.min_distance,
            self.config.interpolation_distance,
        )
        return extract_configured_map(map_graph, scene, self.config)


@functools.lru_cache(maxsize=10)
def _load_argoverse2_map(
    key: str,
    min_distance: float | None,
    interpolation_distance: float | None,
) -> MapGraph:
    return Argoverse2MapBuilder.from_json_file(Path(key)).build(
        min_distance=min_distance,
        interpolation_distance=interpolation_distance,
    )


class Argoverse2Map:
    """Represents a map in the Argoverse2 dataset.

    Parameters
    ----------
    json_file : Path
        Path to the Argoverse 2 JSON map file.
    """

    def __init__(self, json_file: Path) -> None:
        self.json_file: Path = json_file
        with Path.open(json_file, "r") as file:
            self.json_data: dict[str, Any] = json.load(file)

    @cached_property
    def segments(self) -> dict[int, LaneSegment]:
        """Lane segments from the JSON data."""
        segments_data: dict[str, dict[str, Any]] = self.json_data.get("lane_segments", [])
        return {segment["id"]: LaneSegment.from_dict(segment) for segment in segments_data.values()}

    @cached_property
    def pedestrian_crossings(self) -> dict[int, PedestrianCrossing]:
        """Pedestrian crossings from the JSON data."""
        crossings_data: dict[str, dict[str, Any]] = self.json_data.get("pedestrian_crossings", [])
        return {
            crossing["id"]: PedestrianCrossing.from_dict(crossing)
            for crossing in crossings_data.values()
        }


class LaneType(IntEnum):
    """Represents the type of lane segment."""

    VEHICLE = 0
    BIKE = 1
    BUS = 2


class LaneBoundaryType(IntEnum):
    """Represents the type of lane marking."""

    NONE = auto()
    UNKNOWN = auto()
    SOLID_WHITE = auto()
    SOLID_YELLOW = auto()
    DOUBLE_SOLID_WHITE = auto()
    DOUBLE_SOLID_YELLOW = auto()
    DASHED_WHITE = auto()
    DASHED_YELLOW = auto()
    DOUBLE_DASH_WHITE = auto()
    DOUBLE_DASH_YELLOW = auto()
    DASH_SOLID_WHITE = auto()
    DASH_SOLID_YELLOW = auto()
    SOLID_DASH_WHITE = auto()
    SOLID_DASH_YELLOW = auto()
    SOLID_BLUE = auto()


@dataclass
class LaneBoundary:
    """Represents a lane boundary in the Argoverse2 map."""

    lane_type: LaneBoundaryType
    points: list[Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, left: bool) -> LaneBoundary:
        """Create a `LaneBoundary` instance from a dictionary."""
        boundary_key = "left_lane_boundary" if left else "right_lane_boundary"
        type_key = "left_lane_mark_type" if left else "right_lane_mark_type"

        points: list[Point] = [(node["x"], node["y"]) for node in data[boundary_key]]
        return cls(lane_type=LaneBoundaryType[data[type_key]], points=points)

    def get_edge_type(self) -> EdgeType:
        """Get the corresponding `EdgeType` for the lane boundary."""
        return _BOUNDARY_TO_EDGE_TYPE.get(self.lane_type, EdgeType.NONE)


_BOUNDARY_TO_EDGE_TYPE = {
    LaneBoundaryType.NONE: EdgeType.VIRTUAL,
    LaneBoundaryType.UNKNOWN: EdgeType.VIRTUAL,
    LaneBoundaryType.SOLID_WHITE: EdgeType.LINE_THIN,
    LaneBoundaryType.SOLID_YELLOW: EdgeType.LINE_THIN,
    LaneBoundaryType.DOUBLE_SOLID_WHITE: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DOUBLE_SOLID_YELLOW: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DASHED_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DASHED_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DOUBLE_DASH_WHITE: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.DOUBLE_DASH_YELLOW: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.DASH_SOLID_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DASH_SOLID_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_DASH_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_DASH_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_BLUE: EdgeType.LINE_THICK,
}


@dataclass
class LaneSegment:
    """Represents a lane segment in the Argoverse2 map."""

    id: int
    lane_type: LaneType
    left_boundary: LaneBoundary | None = None
    right_boundary: LaneBoundary | None = None
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    right_neighbor_id: int | None = None
    left_neighbor_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaneSegment:
        """Create a `LaneSegment` instance from a dictionary."""
        segment = cls(
            id=data["id"],
            lane_type=LaneType[data["lane_type"]],
            right_neighbor_id=data.get("right_neighbor_id"),
            left_neighbor_id=data.get("left_neighbor_id"),
        )

        if "left_lane_boundary" in data:
            segment.left_boundary = LaneBoundary.from_dict(data, left=True)
        if "right_lane_boundary" in data:
            segment.right_boundary = LaneBoundary.from_dict(data, left=False)

        segment.successors = data.get("successors", [])
        segment.predecessors = data.get("predecessors", [])

        return segment


@dataclass
class PedestrianCrossing:
    """Represents a pedestrian crossing in the Argoverse2 map."""

    id: int
    first_edge: list[Point]
    second_edge: list[Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PedestrianCrossing:
        """Create a `PedestrianCrossing` instance from a dictionary."""
        return cls(
            id=data["id"],
            first_edge=[(node["x"], node["y"]) for node in data["edge1"]],
            second_edge=[(node["x"], node["y"]) for node in data["edge2"]],
        )


class Argoverse2MapBuilder(FeatureMapBuilder):
    """A builder for creating a graph representation of an Argoverse2 map."""

    def __init__(self, map_data: Argoverse2Map) -> None:
        self.map: Argoverse2Map = map_data

    @classmethod
    def from_json_file(cls, json_file: Path) -> Argoverse2MapBuilder:
        """Create a map builder from an Argoverse 2 JSON file."""
        map_data = Argoverse2Map(json_file)
        return cls(map_data)

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        """Yield map features from pedestrian crossings and lane boundaries."""
        for crossing in self.map.pedestrian_crossings.values():
            yield PathFeature(
                points=tuple(crossing.first_edge),
                edge_types=EdgeType.PEDESTRIAN_MARKING,
            )
            yield PathFeature(
                points=tuple(crossing.second_edge),
                edge_types=EdgeType.PEDESTRIAN_MARKING,
            )

        for segment in self.map.segments.values():
            left = self._lane_segment_points(segment, side="left")
            if left is not None:
                points, edge = left
                yield PathFeature(points=tuple(points), edge_types=edge)

            right = self._lane_segment_points(segment, side="right")
            if right is not None:
                points, edge = right
                yield PathFeature(points=tuple(points), edge_types=edge)

    @staticmethod
    def _lane_segment_points(
        segment: LaneSegment,
        side: Literal["left", "right"],
    ) -> tuple[list[Point], EdgeType] | None:
        boundary = segment.left_boundary if side == "left" else segment.right_boundary
        if boundary is None:
            return None

        return boundary.points, boundary.get_edge_type()
