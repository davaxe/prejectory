"""Public map-processing API and semantic map compilation helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from math import ceil
from typing import TYPE_CHECKING, Any, Protocol, override

import numpy as np
import numpy.typing as npt

from prejectory.config.models import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    MapConfig,
    MapEdgeTypeRules,
    MapExtraction,
    SceneExtentExtraction,
    TrajectoryBufferExtraction,
)
from prejectory.core.categories import EdgeType
from prejectory.core.maps import MapGraph
from prejectory.core.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


type Point = tuple[float, float]
"""A 2-D point as `(x, y)`."""


@dataclass(frozen=True, slots=True)
class PointFeature:
    """A standalone map point."""

    point: Point


@dataclass(frozen=True, slots=True)
class PathFeature:
    """A polyline or polygon map feature."""

    points: tuple[Point, ...]
    edge_types: EdgeType | tuple[EdgeType, ...]
    closed: bool = False
    key: str | None = None
    min_distance: float | None = None
    interpolation_distance: float | None = None


type MapFeature = PointFeature | PathFeature
type MapExtractor = Callable[[Scene, MapGraph], MapGraph]


@dataclass(slots=True, frozen=True)
class MapReference:
    """Lightweight scene map reference carried alongside loaded trajectory data."""

    map_key: str | None = None
    """Stable map identifier for the scene, if one is known at ingest time."""
    map_payload: bytes | None = None
    """Serialized map payload already available from trajectory ingestion."""


class MapProvider(Protocol):
    """Resolve map graphs for materialized scenes."""

    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        """Resolve the map for *scene* using loader-supplied map reference data."""
        ...


@dataclass(frozen=True, slots=True)
class BoundMapResolver:
    """Scene-compatible resolver bound to one provider/reference pair."""

    provider: MapProvider
    reference: MapReference

    def __call__(self, scene: Scene) -> MapGraph | None:
        """Resolve the map for *scene* using the bound provider and reference."""
        return self.provider.resolve(scene, self.reference)


def bind_map_provider(provider: MapProvider, reference: MapReference) -> BoundMapResolver:
    """Bind a map provider/reference pair into a `Scene` map resolver."""
    return BoundMapResolver(provider=provider, reference=reference)


def resolve_map_key(scene: Scene, reference: MapReference) -> str | None:
    """Return the effective map key for a scene/reference pair."""
    return reference.map_key or scene.map_key


@dataclass(frozen=True, slots=True)
class SharedMapProvider(MapProvider):
    """Map provider backed by shared-memory map graph names."""

    shared_names: dict[str | None, str] | str
    extractor: MapExtractor | None = None

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = resolve_map_key(scene, reference)
        name = (
            self.shared_names.get(key) if isinstance(self.shared_names, dict) else self.shared_names
        )
        if name is None:
            return None

        with MapGraph.from_shared(name) as map_graph:
            if self.extractor is None:
                return map_graph.copy()

            extracted = self.extractor(scene, map_graph)
            return extracted.copy() if extracted is map_graph else extracted


@dataclass(frozen=True, slots=True)
class SceneBasedMapExtractor:
    """Scene-aware map extractor bound to one extraction config.

    This is intentionally a top-level callable class so that it is pickleable
    by multiprocessing when using the spawn start method.
    """

    extraction: MapExtraction

    def __call__(self, scene: Scene, graph: MapGraph) -> MapGraph:
        """Extract a scene-local subgraph from *graph*."""
        return extract_based_on_scene(graph, scene, self.extraction)


def full_map_extract(_scene: Scene, graph: MapGraph) -> MapGraph:
    """Return the full map graph unchanged.

    This is intentionally a top-level function so that it is pickleable by
    multiprocessing when using the spawn start method.
    """
    return graph


def extract_fn(extraction: MapExtraction) -> MapExtractor:
    """Create a scene-aware map extraction function from config."""
    if isinstance(extraction, FullMapExtraction):
        return full_map_extract

    return SceneBasedMapExtractor(extraction)


def extract_based_on_scene(
    map_graph: MapGraph,
    scene: Scene,
    extraction: MapExtraction,
) -> MapGraph:
    """Extract a subgraph based on the scene and extraction configuration."""
    center_x = scene.frame.select("x").mean().item()
    center_y = scene.frame.select("y").mean().item()
    return extract(
        map_graph,
        center=(center_x, center_y),
        extraction=extraction,
        relevant_positions=scene,
    )


def apply_map_config(map_graph: MapGraph, config: MapConfig) -> MapGraph:
    """Apply config-wide map transforms that do not depend on a scene."""
    return apply_edge_type_config(map_graph, config.edge_types)


def extract_configured_map(map_graph: MapGraph, scene: Scene, config: MapConfig) -> MapGraph:
    """Apply map config and then extract the scene-local subgraph."""
    return extract_based_on_scene(apply_map_config(map_graph, config), scene, config.extraction)


def extract(
    graph: MapGraph,
    center: tuple[float, float] | npt.NDArray[np.floating[Any]] | None,
    extraction: MapExtraction,
    *,
    relevant_positions: npt.NDArray[np.floating[Any]] | Scene | None = None,
) -> MapGraph:
    """Extract a subgraph from a graph according to a map extraction config."""
    if isinstance(relevant_positions, Scene):
        relevant_positions = relevant_positions.frame.select("x", "y").to_numpy()

    match extraction:
        case BoundingBoxExtraction(width=width, height=height):
            return graph.extract_bounding_box(center, width, height)
        case CircularExtraction(radius=radius):
            return graph.extract_radius(center, radius)
        case TrajectoryBufferExtraction(radius=radius):
            if relevant_positions is None:
                msg = "relevant_positions must be provided for TrajectoryBufferExtraction"
                raise ValueError(msg)
            return graph.extract_trajectory_buffer(relevant_positions, radius)
        case SceneExtentExtraction(padding=padding, shape=shape):
            if relevant_positions is None:
                msg = "relevant_positions must be provided for SceneExtentExtraction"
                raise ValueError(msg)
            return graph.extract_extent_for_positions(
                relevant_positions,
                padding,
                use_bbox=shape == "bounding_box",
            )
        case FullMapExtraction():
            return graph


def apply_edge_type_config(map_graph: MapGraph, edge_types: MapEdgeTypeRules | None) -> MapGraph:
    """Apply edge-type remapping and filtering to a graph."""
    if edge_types is None or map_graph.num_edges == 0:
        return map_graph

    include, exclude, remap = _normalize_edge_type_rules(edge_types)
    remapped_edge_types = np.array(map_graph.edge_types, copy=True)
    changed = False
    for source, target in remap.items():
        source_value = int(source)
        target_value = int(target)
        matches = remapped_edge_types == source_value
        if matches.any():
            remapped_edge_types[matches] = target_value
            changed = True

    edge_mask = np.ones(map_graph.num_edges, dtype=bool)
    if include is not None:
        include_values = np.array([int(edge_type) for edge_type in include], dtype=np.int32)
        edge_mask &= np.isin(remapped_edge_types, include_values)
    if exclude:
        exclude_values = np.array([int(edge_type) for edge_type in exclude], dtype=np.int32)
        edge_mask &= ~np.isin(remapped_edge_types, exclude_values)

    if not edge_mask.all():
        filtered_graph = map_graph.filter_edges(edge_mask)
        filtered_edge_types = remapped_edge_types[edge_mask]
        return MapGraph(
            node_positions=filtered_graph.node_positions,
            edge_indices=filtered_graph.edge_indices,
            node_types=filtered_graph.node_types,
            edge_types=filtered_edge_types,
        )

    if not changed:
        return map_graph

    return MapGraph(
        node_positions=map_graph.node_positions,
        edge_indices=map_graph.edge_indices,
        node_types=map_graph.node_types,
        edge_types=remapped_edge_types,
    )


def _normalize_edge_type_rules(
    edge_types: MapEdgeTypeRules,
) -> tuple[frozenset[EdgeType] | None, frozenset[EdgeType], dict[EdgeType, EdgeType]]:
    include = (
        None
        if edge_types.include is None
        else frozenset(EdgeType.from_value(edge_type) for edge_type in edge_types.include)
    )
    exclude = frozenset(EdgeType.from_value(edge_type) for edge_type in edge_types.exclude)
    remap = {
        EdgeType.from_value(source): EdgeType.from_value(target)
        for source, target in edge_types.remap.items()
    }
    return include, exclude, remap


@dataclass(frozen=True, slots=True)
class MapBuildOptions:
    """Global sampling and edge-remapping options for map compilation."""

    min_distance: float
    interpolation_distance: float
    edge_remap: dict[EdgeType, EdgeType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate distance parameters after initialization."""
        if self.interpolation_distance <= 0.0:
            msg = "interpolation_distance must be greater than 0."
            raise ValueError(msg)
        if not (0.0 <= self.min_distance <= self.interpolation_distance):
            msg = (
                "min_distance must be in the range "
                f"[0, interpolation_distance] ([0, {self.interpolation_distance}])."
            )
            raise ValueError(msg)

    @classmethod
    def from_distances(
        cls,
        min_distance: float | None,
        interpolation_distance: float | None,
        *,
        edge_remap: Mapping[EdgeType, EdgeType] | None = None,
    ) -> MapBuildOptions:
        """Create validated options from nullable distance inputs."""
        resolved_min_distance = 0.0 if min_distance is None else min_distance
        resolved_interpolation_distance = (
            np.inf if interpolation_distance is None else interpolation_distance
        )
        return cls(
            min_distance=resolved_min_distance,
            interpolation_distance=resolved_interpolation_distance,
            edge_remap={} if edge_remap is None else dict(edge_remap),
        )


class InterpolationStage(IntEnum):
    """Stage of an interpolation step."""

    INTERMEDIATE = auto()
    LAST = auto()


def interpolate_position(
    src: Point,
    dst: Point,
    target_distance: float | None = None,
) -> list[tuple[InterpolationStage, int, Point]]:
    """Interpolate positions between *src* and *dst*."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    distance = math.sqrt(dx * dx + dy * dy)

    if target_distance is None or target_distance == float("inf") or distance <= target_distance:
        return [(InterpolationStage.LAST, 0, dst)]

    num_steps = ceil(distance / target_distance)
    step_x = dx / num_steps
    step_y = dy / num_steps

    points: list[tuple[InterpolationStage, int, Point]] = []
    for i in range(1, num_steps):
        x = src[0] + step_x * i
        y = src[1] + step_y * i
        points.append((InterpolationStage.INTERMEDIATE, i - 1, (x, y)))

    points.append((InterpolationStage.LAST, num_steps - 1, dst))
    return points


@dataclass(frozen=True, slots=True)
class _CompiledPath:
    start_node: int
    end_node: int


@dataclass(slots=True)
class MapGraphCompiler:
    """Compile semantic map features into a `MapGraph`."""

    options: MapBuildOptions
    _x: list[float] = field(default_factory=list, init=False)
    _y: list[float] = field(default_factory=list, init=False)
    _edge_src: list[int] = field(default_factory=list, init=False)
    _edge_dst: list[int] = field(default_factory=list, init=False)
    _edge_types: list[int] = field(default_factory=list, init=False)
    _seen_edges: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    _paths_by_key: dict[str, _CompiledPath] = field(default_factory=dict, init=False)

    def compile(self, features: Iterable[MapFeature]) -> MapGraph:
        """Compile semantic features into a `MapGraph`."""
        for feature in features:
            if isinstance(feature, PointFeature):
                _ = self._add_node(*feature.point)
            else:
                self._compile_path(feature)

        node_positions = np.column_stack([
            np.array(self._x, dtype=np.float64),
            np.array(self._y, dtype=np.float64),
        ])
        edge_indices = np.array([self._edge_src, self._edge_dst], dtype=np.int32)
        edge_types = np.array(self._edge_types, dtype=np.int32)
        return MapGraph(
            edge_indices=edge_indices,
            node_positions=node_positions,
            edge_types=edge_types,
        )

    def _compile_path(self, feature: PathFeature) -> None:
        points = list(feature.points)
        if not points:
            return

        edge_types = _normalize_edge_types(
            feature.edge_types,
            len(points) - 1 + int(feature.closed),
        )
        retained_points, retained_edge_types = _filter_path_points(
            points=points,
            edge_types=edge_types,
            closed=feature.closed,
            min_distance=self.options.min_distance,
        )
        if not retained_points:
            return

        first_node = self._add_node(*retained_points[0])
        prev_node = first_node
        prev_point = retained_points[0]
        for segment_index, dst_point in enumerate(retained_points[1:]):
            dst_node = self._add_node(*dst_point)
            self._add_interpolated_edge(
                src_id=prev_node,
                src_point=prev_point,
                dst_id=dst_node,
                dst_point=dst_point,
                interpolation_distance=self.options.interpolation_distance,
                edge_type=retained_edge_types[segment_index],
            )
            prev_node = dst_node
            prev_point = dst_point

        end_node = prev_node
        if feature.closed and len(retained_points) > 1:
            self._add_interpolated_edge(
                src_id=prev_node,
                src_point=prev_point,
                dst_id=first_node,
                dst_point=retained_points[0],
                interpolation_distance=self.options.interpolation_distance,
                edge_type=retained_edge_types[-1],
            )

        if feature.key is not None:
            if feature.key in self._paths_by_key:
                msg = f"Duplicate path key {feature.key!r}."
                raise ValueError(msg)
            self._paths_by_key[feature.key] = _CompiledPath(
                start_node=first_node,
                end_node=end_node,
            )

    def _add_interpolated_edge(
        self,
        *,
        src_id: int,
        src_point: Point,
        dst_id: int,
        dst_point: Point,
        interpolation_distance: float,
        edge_type: EdgeType,
    ) -> None:
        prev_id = src_id
        for stage, _, point in interpolate_position(
            src_point,
            dst_point,
            target_distance=interpolation_distance,
        ):
            new_id = dst_id if stage == InterpolationStage.LAST else self._add_node(*point)
            self._add_edge(prev_id, new_id, edge_type)
            prev_id = new_id

    def _add_node(self, x: float, y: float) -> int:
        node_id = len(self._x)
        self._x.append(x)
        self._y.append(y)
        return node_id

    def _add_edge(self, src: int, dst: int, edge_type: EdgeType) -> None:
        edge_type_val = self.options.edge_remap.get(edge_type, edge_type)
        edge_key = (src, dst, int(edge_type_val))
        if edge_key in self._seen_edges:
            return

        self._seen_edges.add(edge_key)
        self._edge_src.append(src)
        self._edge_dst.append(dst)
        self._edge_types.append(int(edge_type_val))


def _normalize_edge_types(
    edge_types: EdgeType | tuple[EdgeType, ...],
    n_edges: int,
) -> list[EdgeType]:
    if isinstance(edge_types, EdgeType):
        return [edge_types] * n_edges

    normalized = list(edge_types)
    if len(normalized) != n_edges:
        msg = (
            "Length of edge_types must equal the number of edges. "
            f"Got {len(normalized)} for {n_edges} edges."
        )
        raise ValueError(msg)
    return normalized


def _filter_path_points(
    *,
    points: list[Point],
    edge_types: list[EdgeType],
    closed: bool,
    min_distance: float,
) -> tuple[list[Point], list[EdgeType]]:
    if len(points) < 2:
        return points[:1], []

    if min_distance <= 0.0:
        retained_points = list(points)
        retained_edge_types = list(edge_types)
        return retained_points, retained_edge_types

    min_dist_sq = min_distance**2
    retained_points = [points[0]]
    retained_edge_types: list[EdgeType] = []
    prev_point = points[0]

    i, j = 0, 1
    while i < len(points) - 1:
        dst_point = points[j]
        if _distance_sq(prev_point, dst_point) < min_dist_sq and j < len(points) - 1:
            j += 1
            continue

        retained_points.append(dst_point)
        retained_edge_types.append(edge_types[i])
        prev_point = dst_point
        i = j
        j = i + 1

    if closed and len(retained_points) > 1:
        retained_edge_types.append(edge_types[-1])

    return retained_points, retained_edge_types


def _distance_sq(src: Point, dst: Point) -> float:
    return (src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2


class FeatureMapBuilder(ABC):
    """Base class for map builders that emit semantic geometry features."""

    @abstractmethod
    def iter_features(self) -> Iterable[MapFeature]:
        """Yield semantic map features to compile."""
        ...

    def edge_remap(self) -> Mapping[EdgeType, EdgeType]:
        """Return optional edge-type remapping for compilation."""
        _ = self
        return {}

    def build(
        self,
        min_distance: float | None = None,
        interpolation_distance: float | None = None,
    ) -> MapGraph:
        """Compile this builder's features into a map graph."""
        options = MapBuildOptions.from_distances(
            min_distance=min_distance,
            interpolation_distance=interpolation_distance,
            edge_remap=self.edge_remap(),
        )
        compiler = MapGraphCompiler(options)
        return compiler.compile(self.iter_features())
