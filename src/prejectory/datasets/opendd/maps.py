"""Map-graph builder for the OpenDD dataset."""

from __future__ import annotations

import functools
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

from typing_extensions import override

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
class OpenDDMapProvider(MapProvider):
    config: MapConfig

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = resolve_map_key(scene, reference)
        if key is None:
            return None

        map_graph = _load_opendd_map(
            str(key),
            self.config.min_distance,
            self.config.interpolation_distance,
        )
        return extract_configured_map(map_graph, scene, self.config)


@functools.lru_cache(maxsize=8)
def _load_opendd_map(
    key: str,
    min_distance: float | None,
    interpolation_distance: float | None,
) -> MapGraph:
    return OpenDDMapBuilder(Path(key)).build(
        min_distance=min_distance,
        interpolation_distance=interpolation_distance,
    )


class OpenDDMapBuilder(FeatureMapBuilder):
    """A builder for creating a graph representation of an OpenDD map."""

    def __init__(self, sqlite_map_path: Path) -> None:
        if not sqlite_map_path.exists():
            msg = f"SQLite file {sqlite_map_path} does not exist."
            raise FileNotFoundError(msg)

        self.connection: sqlite3.Connection = sqlite3.connect(sqlite_map_path)
        self.cursor: sqlite3.Cursor = self.connection.cursor()

    @classmethod
    def from_sqlite_file(cls, sqlite_file: Path | str) -> OpenDDMapBuilder:
        """Create a builder from a SQLite map database path."""
        if isinstance(sqlite_file, str):
            sqlite_file = Path(sqlite_file)
        return cls(sqlite_file)

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        _ = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in self.cursor.fetchall()]
        map_table = table_names[0]
        _ = self.cursor.execute(f"SELECT * FROM {map_table}")  # ruff: ignore[hardcoded-sql-expression]
        rows = self.cursor.fetchall()
        for row in rows:
            map_data_row = MapDataRow.try_from_row(row)
            if map_data_row is None:
                continue
            yield PathFeature(
                points=tuple(map_data_row.geometry.coordinates()),
                edge_types=map_data_row.edge_type(),
                closed=isinstance(map_data_row.geometry, Polygon),
            )


class Geometry(Protocol):
    """Protocol for geometry types."""

    _coordinates: list[Point]

    @classmethod
    def from_str(cls, geometry_str: str) -> Self:
        """Create a Geometry object from a string representation."""
        ...

    def coordinates(self) -> list[Point]:
        """Get the coordinates of the geometry."""
        return self._coordinates


@dataclass
class LineString(Geometry):
    """A LineString geometry type."""

    _coordinates: list[Point]

    @classmethod
    @override
    def from_str(cls, geometry_str: str) -> LineString:
        """Parse a `LINESTRING` geometry from the database representation."""
        if not geometry_str.startswith("LINESTRING"):
            msg = f"Expected 'LINESTRING', got '{geometry_str}'"
            raise ValueError(msg)

        coords_str = geometry_str[len("LINESTRING (") : -1]
        coords = coords_str.split(", ")
        return cls([(float(coord.split()[0]), float(coord.split()[1])) for coord in coords])


@dataclass(slots=True)
class Polygon(Geometry):
    """A Polygon geometry type."""

    _coordinates: list[Point]

    @classmethod
    @override
    def from_str(cls, geometry_str: str) -> Polygon:
        if not geometry_str.startswith("POLYGON"):
            msg = f"Expected 'POLYGON', got '{geometry_str}'"
            raise ValueError(msg)

        coords_str = geometry_str[len("POLYGON ((") : -2]
        coords = coords_str.split(", ")
        points: list[Point] = [
            (float(coord.split()[0]), float(coord.split()[1])) for coord in coords[:-1]
        ]
        return cls(points)


@dataclass
class MapDataRow:
    """A single row of map data from the OpenDD database."""

    index: int
    id: int
    type: str
    geometry: Geometry
    successors: list[int]
    predecessors: list[int]
    marking_type: str | None
    lane_type: str | None
    area_type: str | None

    @classmethod
    def try_from_row(cls, row: tuple[Any, ...]) -> MapDataRow | None:
        """Build a validated map-data row from a SQLite result row."""
        try:
            data = cls(
                row[0],
                row[1],
                row[2],
                MapDataRow._parse_geometry(row[3]),
                MapDataRow._parse_str_list(row[4]),
                MapDataRow._parse_str_list(row[5]),
                row[6],
                row[7],
                row[8],
            )
            if not data._is_valid():
                return None
        except (AttributeError, ValueError):
            return None

        return data

    def edge_type(self) -> EdgeType:
        """Map the row's marking metadata to a graph edge type."""
        if self.marking_type is None:
            return EdgeType.VIRTUAL
        return _MARKING_TYPE_TO_EDGE_TYPE[self.marking_type]

    def _is_valid(self) -> bool:
        return self.area_type not in {"COMPLETE", "NEVER", "EMERGENCY"}

    @staticmethod
    def _parse_str_list(value: str | None) -> list[int]:
        if value == "[]" or value is None:
            return []
        value = value.strip("[]")
        return [int(float(x.strip())) for x in value.split(",")]

    @staticmethod
    def _parse_geometry(geometry_str: str) -> Geometry:
        if geometry_str.startswith("LINESTRING"):
            return LineString.from_str(geometry_str)
        if geometry_str.startswith("POLYGON"):
            return Polygon.from_str(geometry_str)
        msg = f"Unsupported geometry type: {geometry_str}"
        raise ValueError(msg)


_MARKING_TYPE_TO_EDGE_TYPE: dict[str, EdgeType] = {
    "CURB": EdgeType.CURB,
    "CURB_TRAVERSABLE": EdgeType.CURB,
    "SHORT_DASHED_LINE": EdgeType.LINE_THIN_DASHED,
    "SINGLE_SOLID_LINE": EdgeType.LINE_THIN,
    "LONG_DASHED_LINE": EdgeType.LINE_THIN_DASHED,
    "NO_MARKING": EdgeType.VIRTUAL,
    "SHADED_AREA_MARKING": EdgeType.VIRTUAL,
    "GUARDRAIL": EdgeType.GUARD_RAIL,
}
