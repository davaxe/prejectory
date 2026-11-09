"""Map-graph builder for the Waymo Open Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from prejectory.core.categories import EdgeType
from prejectory.datasets.waymo.protos import lean_map_pb2
from prejectory.processing.maps import (
    FeatureMapBuilder,
    MapProvider,
    MapReference,
    PathFeature,
    PointFeature,
    extract_configured_map,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from prejectory.config import MapConfig
    from prejectory.core.maps import MapGraph
    from prejectory.core.scene.model import Scene


@dataclass(frozen=True, slots=True)
class WaymoEmbeddedMapProvider(MapProvider):
    config: MapConfig

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        if reference.map_payload is None:
            return None

        map_data = lean_map_pb2.LeanMapContainer.FromString(reference.map_payload)
        map_graph = WaymoMapBuilder.from_proto(map_data.map_features).build(
            min_distance=self.config.min_distance,
            interpolation_distance=self.config.interpolation_distance,
        )
        return extract_configured_map(map_graph, scene, self.config)


class WaymoMapBuilder(FeatureMapBuilder):
    """A Waymo map representation."""

    def __init__(self) -> None:
        self.road_lines: dict[int, lean_map_pb2.RoadLine] = {}
        self.road_edges: dict[int, lean_map_pb2.RoadEdge] = {}
        self.driveways: dict[int, lean_map_pb2.Driveway] = {}
        self.crosswalks: dict[int, lean_map_pb2.Crosswalk] = {}
        self.lanes: dict[int, lean_map_pb2.LaneCenter] = {}
        self.speed_bumps: dict[int, lean_map_pb2.SpeedBump] = {}
        self.stop_signs: dict[int, lean_map_pb2.StopSign] = {}
        self._no_map_features: bool = True

    @classmethod
    def from_proto(cls, map_features: Sequence[lean_map_pb2.MapFeature]) -> WaymoMapBuilder:
        """Create a Waymo map builder from a list of `MapFeature` protos."""
        waymo_map = cls()
        waymo_map._process_map_features(map_features)
        waymo_map._no_map_features = len(map_features) == 0
        return waymo_map

    @override
    def iter_features(self) -> Iterable[PathFeature | PointFeature]:
        for road_edge in self.road_edges.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in road_edge.polyline),
                edge_types=_ROAD_EDGE_TYPE_TO_EDGE_TYPE[road_edge.type],
            )

        for road_line in self.road_lines.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in road_line.polyline),
                edge_types=_ROAD_LINE_TYPE_TO_EDGE_TYPE[road_line.type],
            )

        for crosswalk in self.crosswalks.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in crosswalk.polygon),
                edge_types=EdgeType.PEDESTRIAN_MARKING,
                closed=True,
                min_distance=0.5,
            )

        for driveway in self.driveways.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in driveway.polygon),
                edge_types=EdgeType.VIRTUAL,
                closed=True,
            )

        for speed_bump in self.speed_bumps.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in speed_bump.polygon),
                edge_types=EdgeType.REGULATORY,
                closed=True,
                min_distance=0.5,
            )

        for lane in self.lanes.values():
            yield PathFeature(
                points=tuple((point.x, point.y) for point in lane.polyline),
                edge_types=EdgeType.VIRTUAL,
            )

        for stop_sign in self.stop_signs.values():
            yield PointFeature(point=(stop_sign.position.x, stop_sign.position.y))

    def _process_map_features(self, map_features: Sequence[lean_map_pb2.MapFeature]) -> None:
        for feature in map_features:
            kind = feature.WhichOneof("feature_data")
            if kind == "road_line":
                self.road_lines[feature.id] = feature.road_line
            elif kind == "road_edge":
                self.road_edges[feature.id] = feature.road_edge
            elif kind == "lane":
                self.lanes[feature.id] = feature.lane
            elif kind == "stop_sign":
                self.stop_signs[feature.id] = feature.stop_sign
            elif kind == "crosswalk":
                self.crosswalks[feature.id] = feature.crosswalk
            elif kind == "speed_bump":
                self.speed_bumps[feature.id] = feature.speed_bump
            elif kind == "driveway":
                self.driveways[feature.id] = feature.driveway


_ROAD_LINE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    lean_map_pb2.RoadLine.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    lean_map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: EdgeType.LINE_THIN_DASHED,
    lean_map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: EdgeType.LINE_THIN_DOUBLE,
    lean_map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: EdgeType.LINE_THIN,
    lean_map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: EdgeType.LINE_THIN_DASHED,
    lean_map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE_DASHED,
    lean_map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: EdgeType.LINE_THIN,
    lean_map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
    lean_map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
}

_ROAD_EDGE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    lean_map_pb2.RoadEdge.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    lean_map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: EdgeType.ROAD_BORDER,
    lean_map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: EdgeType.GUARD_RAIL,
}
