import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import override

from prejectory.core.categories import EdgeType
from prejectory.datasets.shared import utils
from prejectory.processing.maps import FeatureMapBuilder, PathFeature, Point


@dataclass
class OSMWay:
    """Lightweight representation of an OSM Way to replace osmium.Way."""

    tags: dict[str, str]


class OSMMapBuilder(FeatureMapBuilder):
    """Map builder that constructs a `MapGraph` from OpenStreetMap (OSM) XML data."""

    def __init__(
        self,
        osm_file: Path,
        utm_position_offset: tuple[float, float] = (0.0, 0.0),
        edge_type_mapping: Callable[[OSMWay], EdgeType] | None = None,
        *,
        include_edge_type_none: bool = False,
        force_zone_from_origin: tuple[float, float] | None = None,
        local_origin_latlon: tuple[float, float] | None = None,
    ) -> None:
        if not osm_file.exists():
            msg = (
                f"OSM file not found at {osm_file}. "
                "Please provide a valid path to the OSM data file."
            )
            raise FileNotFoundError(msg)

        self._edge_type_mapping: Callable[[OSMWay], EdgeType] = (
            edge_type_mapping or self._default_edge_type_mapping
        )
        self._utm_position_offset: tuple[float, float] = utm_position_offset
        self._osm_file: Path = osm_file
        self._nodes: dict[int, Point] = {}
        self._include_edge_type_none: bool = include_edge_type_none

        self._force_zone_number: int | None = None
        self._force_zone_letter: str | None = None
        self._origin_utm_x: float = 0.0
        self._origin_utm_y: float = 0.0

        if force_zone_from_origin is not None:
            zone_lat, zone_lon = force_zone_from_origin
            _, _, self._force_zone_number, self._force_zone_letter = utils.from_latlon(
                zone_lat,
                zone_lon,
            )

        if local_origin_latlon is not None:
            origin_lat, origin_lon = local_origin_latlon
            self._origin_utm_x, self._origin_utm_y, _, _ = utils.from_latlon(
                origin_lat,
                origin_lon,
                force_zone_number=self._force_zone_number,
                force_zone_letter=self._force_zone_letter,
            )

    @staticmethod
    def _default_edge_type_mapping(way: OSMWay) -> EdgeType:
        return EdgeType.from_str(way.tags.get("type"), way.tags.get("subtype"))

    def _process_node(
        self,
        elem: ET.Element,
        x_offset: float,
        y_offset: float,
        root: ET.Element,
    ) -> None:
        """Process an OSM node element."""
        node_id = int(elem.attrib["id"])
        lat = float(elem.attrib["lat"])
        lon = float(elem.attrib["lon"])

        x, y, _, _ = utils.from_latlon(
            lat,
            lon,
            force_zone_number=self._force_zone_number,
            force_zone_letter=self._force_zone_letter,
        )

        self._nodes[node_id] = (
            x - self._origin_utm_x + x_offset,
            y - self._origin_utm_y + y_offset,
        )

        elem.clear()
        root.clear()

    def _process_way(self, elem: ET.Element, root: ET.Element) -> PathFeature | None:
        """Process an OSM way element."""
        points: list[Point] = []
        tags: dict[str, str] = {}

        for child in elem:
            if child.tag == "nd":
                ref = int(child.attrib["ref"])
                if ref in self._nodes:
                    points.append(self._nodes[ref])
            elif child.tag == "tag":
                tags[child.attrib["k"]] = child.attrib["v"]

        elem.clear()
        root.clear()

        way = OSMWay(tags=tags)
        edge_type = self._edge_type_mapping(way)
        if (edge_type == EdgeType.NONE and not self._include_edge_type_none) or not points:
            return None

        return PathFeature(points=tuple(points), edge_types=edge_type)

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        self._nodes = {}
        context = ET.iterparse(self._osm_file, events=("start", "end"))  # ruff: ignore[suspicious-xml-element-tree-usage]
        iterator = iter(context)
        _, root = next(iterator)

        x_offset, y_offset = self._utm_position_offset
        for event, elem in iterator:
            if event != "end":
                continue
            if elem.tag == "node":
                self._process_node(elem, x_offset, y_offset, root)
            elif elem.tag == "way":
                feature = self._process_way(elem, root)
                if feature is not None:
                    yield feature
