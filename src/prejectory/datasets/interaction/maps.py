"""Map-graph builder for the INTERACTION dataset."""

import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from typing import override

from prejectory.datasets.shared.osm_builder import OSMMapBuilder


class InteractionMapBuilder(OSMMapBuilder):
    """Map builder for the INTERACTION dataset."""

    @override
    def _process_node(
        self,
        elem: ET.Element,
        x_offset: float,
        y_offset: float,
        root: ET.Element,
    ) -> None:
        node_id = int(elem.attrib["id"])
        x = float(elem.attrib["x"])
        y = float(elem.attrib["y"])
        self._nodes[node_id] = (x + x_offset, y + y_offset)
        elem.clear()
        root.clear()
