"""Parser for NuScenes map data.

This module provides a structured interface for parsing and representing
map elements from the NuScenes dataset, such as nodes, lines, polygons,
lanes, road segments, dividers, stop lines, and other traffic-related objects.

It adheres to the NuScenes map schema (version 1.3) and converts JSON-based
map files into strongly typed Python objects for use in further processing and
converting to graph structures.

For details on the map structure, see:
https://www.nuscenes.org/nuscenes?tutorial=maps

NuScenes dataset documentation:
https://www.nuscenes.org/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar

from typing_extensions import override

from prejectory.core.categories import EdgeType
from prejectory.processing.maps import FeatureMapBuilder, PathFeature, Point

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class NuScenesMap:
    """A class representing a NuScenes map, containing various map objects.

    Parameters
    ----------
    json_file : Path
        Path to the JSON file containing the map data.
    """

    def __init__(self, json_file: Path) -> None:
        self.json_file: Path = json_file
        with Path.open(json_file) as f:
            self.json_data: dict[str, Any] = json.load(f)

    @cached_property
    def nodes(self) -> dict[str, Node]:
        """A dictionary of `Node` objects keyed by their str."""
        return _many_from_dict(Node, self.json_data["node"])

    @cached_property
    def lines(self) -> dict[str, Line]:
        """A dictionary of `Line` objects keyed by their str."""
        return _many_from_dict(Line, self.json_data["line"])

    @cached_property
    def polygons(self) -> dict[str, Polygon]:
        """A dictionary of `Polygon` objects keyed by their str."""
        return _many_from_dict(Polygon, self.json_data["polygon"])

    @cached_property
    def road_dividers(self) -> dict[str, RoadDivider]:
        """A dictionary of `RoadDivider` objects keyed by their str."""
        return _many_from_dict(RoadDivider, self.json_data["road_divider"])

    @cached_property
    def road_segments(self) -> dict[str, RoadSegment]:
        """A dictionary of `RoadSegment` objects keyed by their str."""
        return _many_from_dict(RoadSegment, self.json_data["road_segment"])

    @cached_property
    def pedestrian_crossings(self) -> dict[str, PedestrianCrossing]:
        """A dictionary of `PedestrianCrossing` objects keyed by their str."""
        return _many_from_dict(PedestrianCrossing, self.json_data["ped_crossing"])

    @cached_property
    def walkways(self) -> dict[str, Walkway]:
        """A dictionary of `Walkway` objects keyed by their str."""
        return _many_from_dict(Walkway, self.json_data["walkway"])

    @cached_property
    def traffic_lights(self) -> dict[str, TrafficLight]:
        """A dictionary of `TrafficLight` objects keyed by their str."""
        return _many_from_dict(TrafficLight, self.json_data["traffic_light"])

    @cached_property
    def lane_dividers(self) -> dict[str, LaneDivider]:
        """A dictionary of `LaneDivider` objects keyed by their str."""
        return _many_from_dict(LaneDivider, self.json_data["lane_divider"])

    @cached_property
    def stop_lines(self) -> dict[str, StopLine]:
        """A dictionary of `StopLine` objects keyed by their str."""
        return _many_from_dict(StopLine, self.json_data["stop_line"])

    @cached_property
    def lanes(self) -> dict[str, Lane]:
        """A dictionary of `Lane` objects keyed by their str."""
        return _many_from_dict(Lane, self.json_data["lane"])

    @cached_property
    def carpark_areas(self) -> dict[str, CarparkArea]:
        """A dictionary of `Carpark` objects keyed by their str."""
        return _many_from_dict(CarparkArea, self.json_data.get("carpark_area", []))


class StopLineType(IntEnum):
    """Enum representing different types of stop lines in a NuScenes map."""

    TURN_STOP = auto()
    STOP_SIGN = auto()
    PED_CROSSING = auto()
    TRAFFIC_LIGHT = auto()
    YIELD = auto()


class SegmentDividerType(IntEnum):
    """Enum representing different types of segment dividers in a NuScenes map."""

    NIL = auto()
    DOUBLE_DASHED_WHITE = auto()
    SINGLE_SOLID_WHITE = auto()
    SINGLE_SOLID_YELLOW = auto()
    SINGLE_ZIGZAG_WHITE = auto()
    DOUBLE_SOLID_WHITE = auto()

    def to_edge_type(self) -> EdgeType:
        """Convert `SegmentDividerType` to prejectory `EdgeType`."""
        return _SEGMENT_DIVIDER_TYPE_TO_EDGE_TYPE.get(self, EdgeType.VIRTUAL)


_SEGMENT_DIVIDER_TYPE_TO_EDGE_TYPE: dict[SegmentDividerType, EdgeType] = {
    SegmentDividerType.NIL: EdgeType.VIRTUAL,
    SegmentDividerType.SINGLE_SOLID_WHITE: EdgeType.LINE_THIN,
    SegmentDividerType.SINGLE_SOLID_YELLOW: EdgeType.LINE_THIN,
    SegmentDividerType.SINGLE_ZIGZAG_WHITE: EdgeType.REGULATORY,
    SegmentDividerType.DOUBLE_SOLID_WHITE: EdgeType.LINE_THIN_DOUBLE,
    SegmentDividerType.DOUBLE_DASHED_WHITE: EdgeType.LINE_THIN_DOUBLE_DASHED,
}


class LaneType(IntEnum):
    """Enum representing different types of lanes in a NuScenes map."""

    # First two are available in NuScenes. The rest are in View of Delft (VOD)
    NONE = auto()
    CAR = auto()

    # Need explicit value to easily convert from integers in VOD map file
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class _FromDict(Protocol):
    """Protocol for classes that can be created from a dictionary."""

    id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an instance of the class from a dictionary."""
        ...


@dataclass
class Node:
    """A node in the NuScenes map, representing a point in space."""

    id: str
    x: float
    y: float

    def as_point(self) -> tuple[float, float]:
        """Return the node as an `(x, y)` point tuple."""
        return (self.x, self.y)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Create a `Node` instance from a dictionary."""
        return Node(id=str(data["token"]), x=data["x"], y=data["y"])


@dataclass
class Line:
    """A line in the NuScenes map, representing a sequence of nodes."""

    id: str
    nodes: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Line:
        """Create a `Line` instance from a dictionary."""
        return Line(id=str(data["token"]), nodes=[str(node) for node in data["node_tokens"]])


@dataclass
class Polygon:
    """A closed polygon in the NuScenes map, defined by nodes."""

    id: str
    exterior_nodes: list[str]
    holes: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Polygon` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            exterior_nodes=[str(node) for node in data["exterior_node_tokens"]],
            holes=[[str(node) for node in hole] for hole in data.get("interior_node_tokens", [])],
        )


@dataclass
class RoadDivider:
    """A road divider in NuScenes, represented as a line.

    Optionally, the corresponding road segment (as a reference to a
    `RoadSegment` type) can be specified.
    """

    id: str
    line: str
    road_segment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `RoadDivider` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            line=str(data["line_token"]),
            road_segment=str(data.get("road_segment_token", "")) or None,
        )


@dataclass
class RoadSegment:
    """A road segment in NuScenes, defined by a polygon and a list of nodes."""

    id: str
    polygon: str
    is_intersection: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `RoadSegment` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            is_intersection=data.get("is_intersection", False),
        )


@dataclass
class PedestrianCrossing:
    """A pedestrian crossing in NuScenes, represented by a polygon."""

    id: str
    polygon: str
    road_segment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `PedestrianCrossing` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            road_segment=str(data.get("road_segment_token", "")) or None,
        )


@dataclass
class Walkway:
    """A walkway in NuScenes, represented by a polygon."""

    id: str
    polygon: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Walkway` instance from a dictionary."""
        return cls(id=str(data["token"]), polygon=str(data["polygon_token"]))


@dataclass
class TrafficLight:
    """A traffic light in NuScenes, represented by a line."""

    id: str
    line: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `TrafficLight` instance from a dictionary."""
        return cls(id=str(data["token"]), line=str(data["line_token"]))


@dataclass
class LaneDivider:
    """A lane divider in NuScenes, represented by a line and segment types."""

    id: str
    line: str
    segment_types: list[tuple[str, SegmentDividerType]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `LaneDivider` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            line=str(data["line_token"]),
            segment_types=_parse_segment_divider(data.get("lane_divider_segments", [])),
        )


@dataclass
class StopLine:
    """A stop line in NuScenes, represented by a polygon and associated objects."""

    id: str
    polygon: str
    stop_line_type: StopLineType
    pedestrian_crossings: list[str] = field(default_factory=list)
    traffic_lights: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `StopLine` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            stop_line_type=StopLineType[data.get("stop_line_type", "TURN_STOP")],
            pedestrian_crossings=[str(pc) for pc in data.get("ped_crossing_tokens", [])],
            traffic_lights=[str(tl) for tl in data.get("traffic_light_tokens", [])],
        )

    def is_valid(self, *, allow_pedestrian_crossings: bool = False) -> bool:
        """Check if the stop line is valid for use in a map graph.

        Stops lines for pedestrian crossing can cause unwanted clutter in the
        graph, so they can be excluded by setting `allow_pedestrian_crossings`
        to `False`.
        """
        if not (self.traffic_lights or self.pedestrian_crossings):
            return False

        if self.stop_line_type == StopLineType.TURN_STOP:
            return False

        return not (
            self.stop_line_type == StopLineType.PED_CROSSING and not allow_pedestrian_crossings
        )


def _resolve_lane_type(value: str | None) -> LaneType:
    """Resolve a lane type from a string or integer string value.

    Tries the member name first (NuScenes), then the integer value (VOD),
    and falls back to `LaneType.NONE` if neither matches.
    """
    if value is None:
        return LaneType.NONE
    if value in LaneType.__members__:
        return LaneType[value]
    try:
        int_value = int(value)
        if int_value in LaneType._value2member_map_:
            return LaneType(int_value)
    except (ValueError, TypeError):
        pass
    return LaneType.NONE


@dataclass
class Lane:
    """A lane in NuScenes, represented by a polygon and lane dividers."""

    id: str
    polygon: str
    lane_type: LaneType = LaneType.NONE
    left_lane_divider_segments: list[tuple[str, SegmentDividerType]] = field(default_factory=list)
    right_lane_divider_segments: list[tuple[str, SegmentDividerType]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Lane` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            lane_type=_resolve_lane_type(data.get("lane_type")),
            left_lane_divider_segments=_parse_segment_divider(
                data.get("left_lane_divider_segments", []),
            ),
            right_lane_divider_segments=_parse_segment_divider(
                data.get("right_lane_divider_segments", []),
            ),
        )


@dataclass
class CarparkArea:
    """A carpark in NuScenes, represented by a polygon."""

    id: str
    polygon: str
    # Orientation of the parked cars in the carpark area in radians.
    orientation: float = 0.0
    road_block: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `CarparkArea` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            orientation=data.get("orientation", 0.0),
            road_block=str(data.get("road_block_token", "")) or None,
        )


def _parse_segment_divider(segments: list[dict[str, Any]]) -> list[tuple[str, SegmentDividerType]]:
    """Parse a list of segment dividers from a dictionary.

    This is an internal utility method used to convert the
    dictionary representation of segment dividers into a list of tuples
    containing the node str and the segment divider type.
    """
    return [
        (str(segment_dict["node_token"]), SegmentDividerType[segment_dict["segment_type"]])
        for segment_dict in segments
    ]


T = TypeVar("T", bound="_FromDict")


def _many_from_dict(cls: type[T], data: Iterable[dict[str, Any]]) -> dict[str, T]:
    """Deserialize a sequence of dictionaries into a dictionary of objects.

    If an item cannot be deserialized, it is skipped. If `debug` is True, a
    warning is printed for each item that fails to deserialize.

    Parameters
    ----------
    cls : type[T]
        Class to deserialize the items into. Must have a `from_dict`
        class method and an `id` attribute.
    data : Iterable[dict]
        An iterable of dictionaries representing the items to deserialize.

    Returns
    -------
    dict[str, T]
        A dictionary mapping UUIDs to deserialized objects of type `cls`.

    """
    objects: dict[str, T] = {}
    for item in data:
        try:
            obj = cls.from_dict(item)
        except (ValueError, TypeError):
            continue
        objects[obj.id] = obj

    return objects


class NuScenesMapBuilder(FeatureMapBuilder):
    """A builder for creating a MapGraph from a NuscenesMap."""

    def __init__(
        self,
        nuscenes_map: NuScenesMap,
        ignore_edge_types: set[str] | None = None,
        *,
        lane_polygon_edge: EdgeType | None = None,
    ) -> None:
        self.map: NuScenesMap = nuscenes_map
        self.map_nodes: dict[str, Node] = self.map.nodes
        self.lane_polygon_edge: EdgeType | None = lane_polygon_edge
        self.ignore_edge_types: set[str] = set() if ignore_edge_types is None else ignore_edge_types
        self._edge_type_methods: dict[str, Callable[[], Iterable[PathFeature]]] = {
            "road_divider": self._road_divider_features,
            "lane_divider": self._lane_divider_features,
            "walkway": self._walkway_features,
            "pedestrian_crossing": self._pedestrian_crossing_features,
            "traffic_light": self._traffic_light_features,
            "stop_line": self._stop_line_features,
            "lane": self._lane_features,
            "carpark": self._carpark_features,
        }

    @classmethod
    def from_json_file(cls, path: Path, *, ignore_edge_types: set[str] | None = None) -> Self:
        """Create a map builder from a file path."""
        return cls(NuScenesMap(path), ignore_edge_types=ignore_edge_types)

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        for edge_type, method in self._edge_type_methods.items():
            if edge_type in self.ignore_edge_types:
                continue
            yield from method()

    def _node_points(self, node_ids: list[str]) -> list[Point]:
        return [self.map_nodes[i].as_point() for i in node_ids]

    def _road_divider_features(self) -> Iterable[PathFeature]:
        for road_divider in self.map.road_dividers.values():
            line: Line = self.map.lines[road_divider.line]
            yield PathFeature(
                points=tuple(self._node_points(line.nodes)),
                edge_types=EdgeType.LINE_THICK,
            )

    def _lane_divider_features(self) -> Iterable[PathFeature]:
        for lane_divider in self.map.lane_dividers.values():
            points, edge_types = self._extract_edges(lane_divider.segment_types)
            yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))

    def _lane_features(self) -> Iterable[PathFeature]:
        for lane in self.map.lanes.values():
            if lane.left_lane_divider_segments:
                points, edge_types = self._extract_edges(lane.left_lane_divider_segments)
                yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))
            if lane.right_lane_divider_segments:
                points, edge_types = self._extract_edges(lane.right_lane_divider_segments)
                yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))
            if self.lane_polygon_edge is not None:
                lane_polygon: Polygon = self.map.polygons[lane.polygon]
                yield PathFeature(
                    points=tuple(self._node_points(lane_polygon.exterior_nodes)),
                    edge_types=self.lane_polygon_edge,
                    closed=True,
                )

    def _walkway_features(self) -> Iterable[PathFeature]:
        for walkway in self.map.walkways.values():
            polygon: Polygon = self.map.polygons[walkway.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.CURB,
                closed=True,
                min_distance=0.0,
            )

    def _pedestrian_crossing_features(self) -> Iterable[PathFeature]:
        for crossing in self.map.pedestrian_crossings.values():
            polygon: Polygon = self.map.polygons[crossing.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.PEDESTRIAN_MARKING,
                closed=True,
                min_distance=0.0,
            )

    def _traffic_light_features(self) -> Iterable[PathFeature]:
        for traffic_light in self.map.traffic_lights.values():
            line: Line = self.map.lines[traffic_light.line]
            yield PathFeature(
                points=tuple(self._node_points(line.nodes)),
                edge_types=EdgeType.REGULATORY,
                min_distance=0.0,
            )

    def _stop_line_features(self) -> Iterable[PathFeature]:
        for stop_line in self.map.stop_lines.values():
            if not stop_line.is_valid(allow_pedestrian_crossings=False):
                continue
            polygon: Polygon = self.map.polygons[stop_line.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.STOP,
                closed=True,
                min_distance=0.0,
            )

    def _carpark_features(self) -> Iterable[PathFeature]:
        for carpark in self.map.carpark_areas.values():
            polygon: Polygon = self.map.polygons[carpark.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.VIRTUAL,
                closed=True,
                min_distance=0.0,
            )

    def _extract_edges(
        self,
        segments: list[tuple[str, SegmentDividerType]],
    ) -> tuple[list[Point], list[EdgeType]]:
        return (
            [self.map_nodes[n_id].as_point() for n_id, _ in segments],
            [SegmentDividerType.to_edge_type(s_type) for _, s_type in segments[:-1]],
        )
