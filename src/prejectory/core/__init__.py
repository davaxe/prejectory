"""Core scene, map, and shared enum types.

## Import guide

``python
from prejectory.core import AgentCategory, DatasetSplit, MapGraph, Scene
from prejectory.core import CANONICAL, TrajectorySchema, get_trajectory_schema
from prejectory.core import functional
``

This package is the public import surface for the common domain objects used by
readers, runtime planning, and downstream model code.

"""

from prejectory.core import errors
from prejectory.core.categories import (
    AgentCategory,
    AgentCategoryInput,
    AgentCategoryLike,
    DatasetSplit,
    EdgeType,
)
from prejectory.core.maps import MapGraph, SharedMapGraph
from prejectory.core.scene import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_VELOCITY,
    POSITIONS_VELOCITY_ACCELERATION,
    POSITIONS_VELOCITY_YAW,
    POSITIONS_YAW,
    MapResolver,
    Scene,
    TrajectoryField,
    TrajectorySchema,
    available_trajectory_schema_names,
    available_trajectory_schemas,
    get_trajectory_schema,
)

__all__ = [
    "CANONICAL",
    "POSITIONS_ONLY",
    "POSITIONS_VELOCITY",
    "POSITIONS_VELOCITY_ACCELERATION",
    "POSITIONS_VELOCITY_YAW",
    "POSITIONS_YAW",
    "AgentCategory",
    "AgentCategoryInput",
    "AgentCategoryLike",
    "DatasetSplit",
    "EdgeType",
    "MapGraph",
    "MapResolver",
    "Scene",
    "SharedMapGraph",
    "TrajectoryField",
    "TrajectorySchema",
    "available_trajectory_schema_names",
    "available_trajectory_schemas",
    "errors",
    "get_trajectory_schema",
]
