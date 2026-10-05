"""Public map graph API.

This module provides a stable import location for map graph primitives used by
scene and processing APIs.
"""

from __future__ import annotations

import multiprocessing.shared_memory as shm
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.spatial import cKDTree
from typing_extensions import override

from prejectory.core.categories import EdgeType

if TYPE_CHECKING:
    from types import TracebackType

    import numpy.typing as npt

__all__ = ["EdgeType", "MapGraph", "SharedMapGraph"]


class SharedMapGraph:
    """Context manager for accessing a `MapGraph` stored in shared memory.

    Parameters
    ----------
    shared_name : str
        Name of the shared memory block containing the serialized map graph.
    """

    def __init__(self, shared_name: str) -> None:
        self._shared_name: str = shared_name
        self._shared: shm.SharedMemory | None = None
        self._map_graph: MapGraph | None = None

    def open(self) -> MapGraph:
        """Open the shared memory and return a `MapGraph` instance.

        Returns
        -------
        MapGraph
            The `MapGraph` instance loaded from shared memory.

        Raises
        ------
        RuntimeError
            If the shared memory is already open or if there is an issue
            accessing it (e.g., it doesn't exist or has been released).

        """
        if self._shared is not None:
            msg = f"Shared memory with name '{self._shared_name}' is already open."
            raise RuntimeError(msg)

        self._shared = shm.SharedMemory(name=self._shared_name)

        if self._shared.buf is None:
            msg = f"Failed to access shared memory buffer with name '{self._shared_name}'"
            raise RuntimeError(msg)

        # Store the instance so we can release its pointers later
        self._map_graph = MapGraph.from_buffer(self._shared.buf, read_only=True)
        return self._map_graph

    def close(self) -> None:
        """Close the shared memory and release resources.

        Raises
        ------
        RuntimeError
            If the shared memory is not currently open.

        """
        if self._shared is None:
            msg = f"Shared memory with name '{self._shared_name}' is not open."
            raise RuntimeError(msg)

        # Release the MapGraph's references to the shared memory arrays to allow cleanup.
        if self._map_graph is not None:
            self._map_graph.release()
            self._map_graph = None

        self._shared.close()

    def __enter__(self) -> MapGraph:
        """Access the `MapGraph` from shared memory."""
        return self.open()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_t: TracebackType | None,
    ) -> None:
        """Release resources when exiting the context."""
        self.close()

    def __del__(self) -> None:
        """Ensure shared memory is closed."""
        # Ensure shared memory is closed if the context manager was not used
        # properly.
        if self._shared is not None:
            self._shared.close()

        if self._map_graph is not None:
            self._map_graph.release()

    def copy(self, *, open_if_needed: bool = False) -> MapGraph:
        """Create a deep copy of the `MapGraph` data.

        This is useful if you want to modify the graph data without affecting
        the shared memory version.

        Returns
        -------
        MapGraph
            A new `MapGraph` instance with copied data.

        """
        if self._map_graph is None:
            if open_if_needed:
                self._map_graph = self.open()
            else:
                msg = "Shared memory graph is not loaded. Cannot clone."
                raise RuntimeError(msg)

        return self._map_graph.copy()


@dataclass(init=False, repr=False, slots=True)
class MapGraph:
    """Represents a graph structure for a map, including nodes and edges.

    Parameters
    ----------
    node_positions : ndarray of float64, shape (N, 2)
        Array of node positions.
    edge_indices : ndarray of int64, shape (2, M)
        Array of edge endpoint indices.
    node_types : ndarray of int64, shape (N,), optional
        Per-node type labels. If None, all nodes are assigned type 1.
    edge_types : ndarray of int64, shape (M,), optional
        Per-edge type labels. If None, all edges are assigned type 1.
    """

    node_positions: npt.NDArray[np.float64]
    """Node positions, shape `(N, 2)`."""

    edge_indices: npt.NDArray[np.int32]
    """Edge COO indices, shape `(2, M)`."""

    node_types: npt.NDArray[np.int32]
    """Per-node type labels, shape `(N,)`."""

    edge_types: npt.NDArray[np.int32]
    """Per-edge type labels, shape `(M,)`."""

    def __init__(
        self,
        node_positions: npt.NDArray[np.float64],
        edge_indices: npt.NDArray[np.int32],
        node_types: npt.NDArray[np.int32] | None = None,
        edge_types: npt.NDArray[np.int32] | None = None,
    ) -> None:
        if node_positions.size == 0 and edge_indices.size == 0:
            node_positions = np.zeros((0, 2), dtype=np.float64)
            edge_indices = np.zeros((2, 0), dtype=np.int32)
            node_types = np.zeros((0,), dtype=np.int32)
            edge_types = np.zeros((0,), dtype=np.int32)

        self.node_positions = np.ascontiguousarray(node_positions, dtype=np.float64)
        self.edge_indices = np.ascontiguousarray(edge_indices, dtype=np.int32)

        self.node_types = (
            np.ascontiguousarray(node_types, dtype=np.int32)
            if node_types is not None
            else np.ones(self.num_nodes, dtype=np.int32)
        )

        self.edge_types = (
            np.ascontiguousarray(edge_types, dtype=np.int32)
            if edge_types is not None
            else np.ones(self.num_edges, dtype=np.int32)
        )
        self._validate()

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return int(self.node_positions.shape[0])

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return int(self.edge_indices.shape[1])

    def _validate(self) -> None:
        if self.node_positions.ndim != 2 or self.node_positions.shape[1:] != (2,):
            msg = f"node_positions must have shape (N, 2), got {self.node_positions.shape!r}"
            raise ValueError(msg)
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            msg = f"edge_indices must have shape (2, E), got {self.edge_indices.shape!r}"
            raise ValueError(msg)
        if self.node_types.shape != (self.num_nodes,):
            msg = "node_types length must match the number of nodes"
            raise ValueError(msg)
        if self.edge_types.shape != (self.num_edges,):
            msg = "edge_types length must match the number of edges"
            raise ValueError(msg)
        if not np.isfinite(self.node_positions).all():
            msg = "node_positions must contain only finite values"
            raise ValueError(msg)
        if self.num_edges:
            if self.num_nodes == 0:
                msg = "edges cannot exist without nodes"
                raise ValueError(msg)
            if int(self.edge_indices.min()) < 0 or int(self.edge_indices.max()) >= self.num_nodes:
                msg = "edge_indices contain node indices outside the graph"
                raise ValueError(msg)

    def to_shared(self) -> shm.SharedMemory:
        """Serialize the `MapGraph` to a shared memory block, including dimensions.

        Returns
        -------
        SharedMemory
            `SharedMemory` object containing the serialized graph data.

        """
        # Pack the dimensions into an int64 array to serve as the header
        header = np.array([self.num_nodes, self.num_edges], dtype=np.int64)

        arrays_to_serialize = (
            header,
            self.node_positions,
            self.edge_indices,
            self.node_types,
            self.edge_types,
        )

        total_bytes = sum(arr.nbytes for arr in arrays_to_serialize)
        shared = shm.SharedMemory(create=True, size=total_bytes)

        offset = 0
        for arr in arrays_to_serialize:
            target_arr: npt.NDArray[Any] = np.ndarray(
                arr.shape,
                dtype=arr.dtype,
                buffer=shared.buf,
                offset=offset,
            )
            np.copyto(target_arr, arr)
            offset += arr.nbytes

        return shared

    @classmethod
    def from_shared(cls, shared_name: str) -> SharedMapGraph:
        """Create a context manager for accessing a graph in shared memory.

        Parameters
        ----------
        shared_name : str
            Name of the shared memory block containing the graph data.

        """
        return SharedMapGraph(shared_name)

    @classmethod
    def from_buffer(cls, buf: memoryview, *, read_only: bool = True) -> Self:
        """Deserialize a `MapGraph` from a bytes buffer with embedded metadata."""
        offset = 0

        # Read the first two int64 values to get the dimensions
        header = np.frombuffer(buf, dtype=np.int64, count=2, offset=offset)
        n, m = int(header[0]), int(header[1])
        offset += header.nbytes

        positions = np.frombuffer(buf, dtype=np.float64, count=n * 2, offset=offset).reshape((n, 2))
        offset += positions.nbytes

        indices = np.frombuffer(buf, dtype=np.int32, count=m * 2, offset=offset).reshape((2, m))
        offset += indices.nbytes

        node_types = np.frombuffer(buf, dtype=np.int32, count=n, offset=offset)
        offset += node_types.nbytes

        edge_types = np.frombuffer(buf, dtype=np.int32, count=m, offset=offset)

        if read_only:
            for arr in (positions, indices, node_types, edge_types):
                arr.flags.writeable = False

        return cls(
            node_positions=positions,
            edge_indices=indices,
            node_types=node_types,
            edge_types=edge_types,
        )

    def release(self) -> None:
        """Release references to the internal arrays to allow shared memory cleanup.

        This will clear the internal arrays and is not intended for general use.
        """
        self.node_positions = np.array([], dtype=np.float64).reshape(0, 2)
        self.edge_indices = np.array([], dtype=np.int32).reshape(2, 0)
        self.node_types = np.array([], dtype=np.int32)
        self.edge_types = np.array([], dtype=np.int32)

    def copy(self) -> MapGraph:
        """Create a deep copy of the `MapGraph` instance.

        Returns
        -------
        MapGraph
            A new `MapGraph` instance with copied data.

        """
        return MapGraph(
            node_positions=np.copy(self.node_positions),
            edge_indices=np.copy(self.edge_indices),
            node_types=np.copy(self.node_types),
            edge_types=np.copy(self.edge_types),
        )

    def extract_radius(
        self,
        center: tuple[float, float] | npt.NDArray[np.floating[Any]] | None,
        radius: float,
    ) -> MapGraph:
        """Extract a subgraph within a specified radius from a center point.

        Parameters
        ----------
        center : tuple[float, float] or ndarray
            Center of the radius as a tuple `(x, y)` or array.
        radius : float
            Radius within which to extract nodes and edges.

        Returns
        -------
        MapGraph
            A new `MapGraph` containing only the nodes and edges within the
            given radius.

        """
        if center is None:
            center = np.mean(self.node_positions, axis=0)

        center_arr = np.asarray(center, dtype=np.float64)

        # Squared-distance avoids the sqrt
        diff = self.node_positions - center_arr
        dist_sq = np.sum(diff * diff, axis=1)
        within_mask: npt.NDArray[np.bool_] = dist_sq <= radius * radius

        return self._subgraph_from_mask(within_mask)

    def extract_bounding_box(
        self,
        center: tuple[float, float] | npt.NDArray[np.floating[Any]] | None,
        width: float,
        height: float,
    ) -> MapGraph:
        """Extract a subgraph within a bounding box.

        Parameters
        ----------
        center : tuple[float, float] or ndarray or None
            Center of the bounding box as a tuple `(x, y)` or array. If
            `None`, the mean position of all nodes is used.
        width : float
            Full width of the bounding box.
        height : float
            Full height of the bounding box.

        Returns
        -------
        MapGraph
            A new `MapGraph` with only the nodes and edges inside the box.

        """
        if center is None:
            center_arr = np.mean(self.node_positions, axis=0)
        else:
            center_arr = np.asarray(center, dtype=np.float64)

        half_w = width / 2.0
        half_h = height / 2.0

        x = self.node_positions[:, 0]
        y = self.node_positions[:, 1]

        within_mask: npt.NDArray[np.bool_] = (
            (x >= center_arr[0] - half_w)
            & (x <= center_arr[0] + half_w)
            & (y >= center_arr[1] - half_h)
            & (y <= center_arr[1] + half_h)
        )

        return self._subgraph_from_mask(within_mask)

    def extract_extent_for_positions(
        self,
        relevant_positions: npt.NDArray[np.floating[Any]],
        padding_factor: float = 1.0,
        *,
        use_bbox: bool = False,
    ) -> MapGraph:
        """Extract to cover relevant positions.

        Parameters
        ----------
        relevant_positions : npt.NDArray[np.floating[Any]]
            The positions of relevant positions (not nodes).
        padding_factor : float, optional
            Factor to pad the extracted area, by default 1.0 (no padding).
        use_bbox : bool, optional
            Whether to use a bounding box instead of a radius, by default False

        Returns
        -------
        MapGraph
            The extracted subgraph.

        """
        if not use_bbox:
            center = relevant_positions.mean(axis=0)
            distances = np.linalg.norm(relevant_positions - center, axis=1)
            radius = distances.max() * padding_factor
            return self.extract_radius(center, radius)

        mins = relevant_positions.min(axis=0)
        maxs = relevant_positions.max(axis=0)
        center = (mins + maxs) / 2
        half_w = (maxs[0] - mins[0]) * padding_factor
        half_h = (maxs[1] - mins[1]) * padding_factor

        return self.extract_bounding_box(center, half_w, half_h)

    def extract_trajectory_buffer(
        self,
        relevant_positions: npt.NDArray[np.floating[Any]],
        radius: float,
    ) -> MapGraph:
        """Extract nodes within a fixed radius of any relevant position.

        Parameters
        ----------
        relevant_positions : npt.NDArray[np.floating[Any]]
            Positions that define the buffered trajectory support.
        radius : float
            Euclidean buffer radius around each relevant position.

        Returns
        -------
        MapGraph
            The extracted subgraph.

        """
        if relevant_positions.size == 0:
            return MapGraph(
                node_positions=np.zeros((0, 2), dtype=np.float64),
                edge_indices=np.zeros((2, 0), dtype=np.int32),
                node_types=np.zeros((0,), dtype=np.int32),
                edge_types=np.zeros((0,), dtype=np.int32),
            )

        positions = np.asarray(relevant_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            msg = f"relevant_positions must have shape (N, 2), got {positions.shape!r}"
            raise ValueError(msg)

        if self.num_nodes == 0:
            return self.copy()

        within_mask = np.zeros(self.num_nodes, dtype=bool)
        tree = cKDTree(self.node_positions)
        neighbor_groups = tree.query_ball_point(positions, r=radius)
        for node_indices in neighbor_groups:
            within_mask[node_indices] = True

        return self._subgraph_from_mask(within_mask)

    def filter_edges(
        self,
        edge_mask: npt.NDArray[np.bool_],
        *,
        prune_unreferenced_nodes: bool = True,
    ) -> MapGraph:
        """Return a graph with only the selected edges.

        Parameters
        ----------
        edge_mask : ndarray of bool, shape `(E,)`
            Boolean mask selecting which edges to keep.
        prune_unreferenced_nodes : bool, optional
            Whether to drop nodes that are no longer referenced by any kept
            edge. Defaults to `True`.

        Returns
        -------
        MapGraph
            A new graph containing only the selected edges.

        """
        mask = np.asarray(edge_mask, dtype=bool)
        if mask.shape != (self.num_edges,):
            msg = f"edge_mask must have shape ({self.num_edges},), got {mask.shape!r}"
            raise ValueError(msg)

        if self.num_edges == 0 or mask.all():
            return self.copy()

        kept_edge_indices = self.edge_indices[:, mask]
        kept_edge_types = self.edge_types[mask]

        if not prune_unreferenced_nodes:
            return MapGraph(
                node_positions=self.node_positions,
                edge_indices=kept_edge_indices,
                node_types=self.node_types,
                edge_types=kept_edge_types,
            )

        if kept_edge_indices.size == 0:
            return MapGraph(
                node_positions=np.zeros((0, 2), dtype=np.float64),
                edge_indices=np.zeros((2, 0), dtype=np.int32),
                node_types=np.zeros((0,), dtype=np.int32),
                edge_types=np.zeros((0,), dtype=np.int32),
            )

        node_mask = np.zeros(self.num_nodes, dtype=bool)
        node_mask[np.unique(kept_edge_indices)] = True
        remap = np.full(self.num_nodes, -1, dtype=np.int32)
        remap[node_mask] = np.arange(node_mask.sum(), dtype=np.int32)

        return MapGraph(
            node_positions=self.node_positions[node_mask],
            edge_indices=remap[kept_edge_indices],
            node_types=self.node_types[node_mask],
            edge_types=kept_edge_types,
        )

    def _subgraph_from_mask(self, node_mask: npt.NDArray[np.bool_]) -> MapGraph:
        """Build a new `MapGraph` from a boolean node mask."""
        new_edge_indices, edge_mask = _extract_subgraph(node_mask, self.edge_indices)

        return MapGraph(
            node_positions=self.node_positions[node_mask],
            edge_indices=new_edge_indices,
            node_types=self.node_types[node_mask],
            edge_types=self.edge_types[edge_mask],
        )

    @override
    def __str__(self) -> str:
        """Return a string representation of the MapGraph."""
        return (
            f"MapGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"node_positions_shape={self.node_positions.shape}, "
            f"edge_indices_shape={self.edge_indices.shape})"
        )

    @override
    def __repr__(self) -> str:
        """Return a detailed string representation of the MapGraph."""
        return (
            f"MapGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"node_positions={self.node_positions!r}, "
            f"edge_indices={self.edge_indices!r}, "
            f"node_types={self.node_types!r}, "
            f"edge_types={self.edge_types!r})"
        )


def _extract_subgraph(
    node_mask: npt.NDArray[np.bool_],
    edge_indices: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    """Extract a subgraph given a boolean node mask.

    This is a pure-NumPy replacement for `torch_geometric.utils.subgraph`. It
    filters edges so that both endpoints belong to the selected node subset and
    relabels node indices to be contiguous (0 … K-1).

    Parameters
    ----------
    node_mask : ndarray of bool, shape (N,)
        Boolean array indicating which nodes to keep.
    edge_indices : ndarray of int64, shape (2, M)
        DatasetSource/destination indices for every edge.

    Returns
    -------
    new_edge_indices : ndarray of int64, shape (2, M')
        Relabelled edge index array for the kept edges.
    edge_mask : ndarray of bool, shape (M,)
        Boolean array indicating which edges were kept.

    """
    src = edge_indices[0]
    dst = edge_indices[1]

    # Keep only edges where *both* endpoints are in the subset
    edge_mask: npt.NDArray[np.bool_] = node_mask[src] & node_mask[dst]

    # Build a mapping from old node index → new contiguous index.
    # Positions where node_mask is False get -1 (unused).
    remap = np.full(node_mask.shape[0], -1, dtype=np.int32)
    remap[node_mask] = np.arange(node_mask.sum(), dtype=np.int32)

    kept_src = remap[src[edge_mask]]
    kept_dst = remap[dst[edge_mask]]

    new_edge_indices = np.stack([kept_src, kept_dst], axis=0)
    return new_edge_indices, edge_mask
