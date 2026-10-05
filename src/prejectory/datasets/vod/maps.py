"""Map-graph builder for the View-of-Delft dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from typing_extensions import override

from prejectory.core.categories import EdgeType
from prejectory.datasets.nuscenes.maps import NuScenesMap, NuScenesMapBuilder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from prejectory.processing.maps import PathFeature


class VodMapBuilder(NuScenesMapBuilder):
    """A builder for creating a MapGraph from a VOD map."""

    def __init__(self, path: Path, *, ignore_edge_types: set[str] | None = None) -> None:
        super().__init__(
            NuScenesMap(path),
            ignore_edge_types=ignore_edge_types,
            lane_polygon_edge=EdgeType.LINE_THIN,
        )
        self._edge_type_methods: dict[str, Callable[[], Iterable[PathFeature]]] = {
            "road_divider": self._road_divider_features,
            "walkway": self._walkway_features,
            "pedestrian_crossing": self._pedestrian_crossing_features,
            "traffic_light": self._traffic_light_features,
            "stop_line": self._stop_line_features,
            "lane": self._lane_features,
            "carpark": self._carpark_features,
        }

    @override
    @classmethod
    def from_json_file(cls, path: Path, *, ignore_edge_types: set[str] | None = None) -> Self:
        """Create a VOD map builder from a file path."""
        return cls(path, ignore_edge_types=ignore_edge_types)
