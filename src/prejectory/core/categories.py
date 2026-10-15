"""Enumerations and coercion helpers for shared categorical types."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from enum import IntEnum, StrEnum
from typing import Final, TypeVar


class EdgeType(IntEnum):
    """Enumeration of edge types in the lane graph."""

    NONE = 1
    ROAD_BORDER = 2
    CURB = 3
    REGULATORY = 4
    VIRTUAL = 5
    LINE_THIN = 6
    LINE_THIN_DASHED = 7
    LINE_THICK = 8
    LINE_THICK_DASHED = 9
    PEDESTRIAN_MARKING = 10
    BIKE_MARKING = 11
    GUARD_RAIL = 12
    STOP = 13
    LINE_THIN_DOUBLE = 14
    LINE_THIN_DOUBLE_DASHED = 15

    @classmethod
    def from_str(cls, type_str: str | None, subtype: str | None = None) -> EdgeType:
        """Convert string representation to EdgeType."""
        if type_str is None:
            return cls.NONE

        # First check high-priority mappings
        type_map: dict[str, EdgeType] = _STR_TO_EDGE

        # Check if it's a high-priority type first
        edge_type = type_map.get(type_str)
        if edge_type is not None:
            return edge_type

        # Only then check for line types
        if type_str == "line_thin":
            return cls.LINE_THIN_DASHED if subtype == "dashed" else cls.LINE_THIN
        if type_str == "line_thick":
            return cls.LINE_THICK_DASHED if subtype == "dashed" else cls.LINE_THICK

        return cls.NONE

    @classmethod
    def from_string(cls, value: str) -> EdgeType:
        """Convert a string to an `EdgeType`, case-insensitive."""
        normalized = value.strip().lower().replace("-", "_")
        for edge_type in cls:
            if edge_type.name.lower() == normalized:
                return edge_type

        if normalized in _STR_TO_EDGE:
            return _STR_TO_EDGE[normalized]
        if normalized == "line_thin":
            return cls.LINE_THIN
        if normalized == "line_thick":
            return cls.LINE_THICK

        msg = f"Unknown edge type: {value}"
        raise ValueError(msg)

    @classmethod
    def from_value(cls, value: int | str | EdgeType) -> EdgeType:
        """Convert an integer or string to an `EdgeType`."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls.from_string(value)


class DatasetSplit(StrEnum):
    """Enum representing the available dataset splits."""

    TRAIN = "train"
    """Training split partition, used for model training."""
    VAL = "val"
    """Validation split partition, used for model validation."""
    TEST = "test"
    """Test split partition, used for final model evaluation."""


class AgentCategory(IntEnum):
    """Enumeration of categories of agents / objects.

    !!! note "Dataset-dependent categories"
        Specific categories differ across datasets, and are mapped
        from their original dataset-specific labels into this shared set of
        categories as closely as possible.

    """

    ANIMAL = 1
    """Non-human agents, including pets and wildlife."""
    BICYCLE = 2
    """Two-wheeled non-motorized vehicles."""
    BUS = 3
    """Large passenger vehicles, including city buses."""
    CAR = 4
    """Passenger vehicles, including sedans, SUVs, and pickup trucks."""
    EMERGENCY_VEHICLE = 5
    """Emergency response vehicles."""
    MOTORCYCLE = 6
    """Two-wheeled motorized vehicles."""
    MOVEABLE_OBJECT = 7
    """Generic category for moveable-objects (dataset dependent)."""
    PEDESTRIAN = 8
    """People walking or running."""
    STATIC_OBJECT = 9
    """Generic category for static-objects (dataset dependent)."""
    TRAILER = 10
    """Semi-trailers and other trailer types."""
    TRAM = 11
    """Street-level rail vehicles."""
    TRICYCLE = 12
    """Three-wheeled vehicles."""
    TRUCK = 13
    """Heavy-duty vehicles."""
    UNIMPORTANT = 14
    """Agents that are unimportant to model for a given dataset or task."""
    UNKNOWN = 15
    """Agents of unknown or unclassifiable category."""
    VAN = 16
    """Larger passenger vehicles, including minivans and cargo vans."""

    @classmethod
    def from_string(cls, value: str) -> AgentCategory:
        """Convert a string to an AgentCategory, case-insensitive."""
        s: str = value.lower()
        for category in AgentCategory:
            if category.name.lower() == s:
                return category
        msg = f"Unknown agent category: {s}"
        raise ValueError(msg)

    @classmethod
    def from_value(cls, value: int | str | AgentCategory) -> AgentCategory:
        """Convert an integer or string to an AgentCategory."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls.from_string(value)


AgentCategoryLike = int | str | AgentCategory
"""Type for specifying a single agent category."""
AgentCategoryInput = AgentCategoryLike | Iterable[AgentCategoryLike]
"""Input type for specifying one or more agent categories."""
EdgeTypeLike = int | str | EdgeType
"""Type for specifying a single edge type."""
EdgeTypeInput = EdgeTypeLike | Iterable[EdgeTypeLike]
"""Input type for specifying one or more edge types."""

_AgentCollectionT = TypeVar("_AgentCollectionT", bound=Collection[AgentCategory])
_EdgeCollectionT = TypeVar("_EdgeCollectionT", bound=Collection[EdgeType])


def coerce_agent_categories(
    value: AgentCategoryInput,
    collection: Callable[[Iterable[AgentCategory]], _AgentCollectionT],
) -> _AgentCollectionT:
    """Convert one or many agent-category values into a normalized collection."""
    values = [value] if isinstance(value, (str, int, AgentCategory)) else value
    return collection(AgentCategory.from_value(item) for item in values)


def coerce_edge_types(
    value: EdgeTypeInput,
    collection: Callable[[Iterable[EdgeType]], _EdgeCollectionT],
) -> _EdgeCollectionT:
    """Convert one or many edge-type values into a normalized collection."""
    values = [value] if isinstance(value, (str, int, EdgeType)) else value
    return collection(EdgeType.from_value(item) for item in values)


_STR_TO_EDGE: Final[dict[str, EdgeType]] = {
    "road_border": EdgeType.ROAD_BORDER,
    "curbstone": EdgeType.CURB,
    "stop_line": EdgeType.STOP,
    "regulatory_element": EdgeType.REGULATORY,
    "virtual": EdgeType.VIRTUAL,
    "pedestrian_marking": EdgeType.PEDESTRIAN_MARKING,
    "bike_marking": EdgeType.BIKE_MARKING,
    "guard_rail": EdgeType.GUARD_RAIL,
    "fence": EdgeType.ROAD_BORDER,
    "wall": EdgeType.ROAD_BORDER,
}
