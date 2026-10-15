"""PyTorch Geometric adapters built on top of generic Torch scene datasets.

Ego metadata is batch-safe: `ego_agent_id` uses -1 when absent and
`has_ego_agent_id` indicates whether the source provided an ID.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy as shallow_copy
from typing import TYPE_CHECKING, Generic

from typing_extensions import override

from prejectory.core.optional import raise_missing_optional_dependency
from prejectory.io.adapters.torch import IterableTorchForecastDataset, IterableTorchSceneDataset

try:
    from torch.utils.data import Dataset, IterableDataset
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.data import Dataset as PyGDataset

    from prejectory.io.adapters.torch import (
        IterableReaderT,
        ReaderT,
        TorchForecastDataset,
        TorchForecastRecord,
        TorchSceneDataset,
        TorchSceneRecord,
    )
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The PyG scene dataset adapter", extra="pyg")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch
    from torch_geometric.data.dataset import BaseData

    from prejectory.io.records import PredictionBounds

HeteroDataTransform = Callable[[HeteroData], HeteroData]


class HeteroSceneDataset(PyGDataset, Dataset[HeteroData], Generic[ReaderT]):
    """PyG dataset view over full-horizon Prejectory scene records.

    Each record is a `HeteroData` object with `agent` and `map` node stores and
    a `("map", "connects", "map")` edge store. Agent trajectories are exposed
    as `agent.features` with a matching `agent.valid_mask`.
    """

    def __init__(
        self,
        reader: ReaderT,
        *,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.dataset: TorchSceneDataset[ReaderT] = TorchSceneDataset(reader, copy=copy)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one record converted to `HeteroData`."""
        return _convert_full_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        """Return the number of records visible through the wrapped dataset."""
        return len(self.dataset)


class IterableHeteroSceneDataset(IterableDataset[HeteroData], Generic[IterableReaderT]):
    """Iterable PyG dataset view over full-horizon Prejectory scene records."""

    @override
    def __init__(
        self,
        reader: IterableReaderT,
        *,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__()
        self._transform: HeteroDataTransform | None = transform
        self.dataset: IterableTorchSceneDataset[IterableReaderT] = IterableTorchSceneDataset(
            reader,
            copy=copy,
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding records converted to `HeteroData`."""
        for record in self.dataset:
            hetero = _convert_full_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    def __len__(self) -> int:
        """Return the number of records visible through the wrapped dataset."""
        return len(self.dataset)


class HeteroForecastDataset(PyGDataset, Dataset[HeteroData], Generic[ReaderT]):
    """PyG dataset view with explicit history and future agent tensors."""

    def __init__(
        self,
        reader: ReaderT,
        *,
        bounds: PredictionBounds | None = None,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.dataset: TorchForecastDataset[ReaderT] = TorchForecastDataset(
            reader,
            bounds=bounds,
            copy=copy,
        )

    @override
    def get(self, idx: int) -> HeteroData:
        return _convert_forecast_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        return len(self.dataset)


class IterableHeteroForecastDataset(IterableDataset[HeteroData], Generic[IterableReaderT]):
    """Iterable PyG forecast view over full-horizon scene records."""

    def __init__(
        self,
        reader: IterableReaderT,
        *,
        bounds: PredictionBounds | None = None,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__()
        self._transform: HeteroDataTransform | None = transform
        self.dataset: IterableTorchForecastDataset[IterableReaderT] = IterableTorchForecastDataset(
            reader,
            bounds=bounds,
            copy=copy,
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        for record in self.dataset:
            hetero = _convert_forecast_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    def __len__(self) -> int:
        """Report the number of records exposed by the wrapped reader."""
        return len(self.dataset)


def collate_hetero_with_time_padding(records: Sequence[HeteroData]) -> Batch:
    """Batch hetero scenes by padding agent time axes within the current batch."""
    if not records:
        msg = "`records` must contain at least one HeteroData object."
        raise ValueError(msg)

    max_horizon_frames = max(int(record["agent"].features.size(1)) for record in records)
    padded_records: list[BaseData] = [
        _pad_full_hetero_time_axes(record, horizon_frames=max_horizon_frames) for record in records
    ]
    return Batch.from_data_list(padded_records)


def collate_forecast_hetero_with_time_padding(records: Sequence[HeteroData]) -> Batch:
    """Batch forecast scenes with aligned, independently padded time axes."""
    if not records:
        msg = "`records` must contain at least one HeteroData object."
        raise ValueError(msg)
    max_history = max(int(record["agent"].history_features.size(1)) for record in records)
    max_future = max(int(record["agent"].future_features.size(1)) for record in records)
    padded_records: list[BaseData] = [
        _pad_forecast_hetero_time_axes(record, history_frames=max_history, future_frames=max_future)
        for record in records
    ]
    return Batch.from_data_list(padded_records)


def _convert_full_to_hetero(record: TorchSceneRecord) -> HeteroData:
    data = _hetero_with_common_data(record, num_agents=record.features.size(0))
    data["agent"].features = record.features
    data["agent"].valid_mask = record.valid_mask
    return data


def _convert_forecast_to_hetero(record: TorchForecastRecord) -> HeteroData:
    data = _hetero_with_common_data(record, num_agents=record.history_features.size(0))
    data["agent"].history_features = record.history_features
    data["agent"].history_mask = record.history_mask
    data["agent"].future_features = record.future_features
    data["agent"].future_mask = record.future_mask
    return data


def _hetero_with_common_data(
    record: TorchSceneRecord | TorchForecastRecord,
    *,
    num_agents: int,
) -> HeteroData:
    data = HeteroData()
    data["agent"].agent_id = record.agent_ids
    data["agent"].agent_type = record.agent_types
    data["agent"].screened_agent_mask = record.screened_agent_mask
    data["agent"].num_nodes = num_agents
    data["map"].x = record.map_node_positions
    data["map"].node_type = record.map_node_types
    data["map"].num_nodes = record.map_node_positions.size(0)
    data["map", "connects", "map"].edge_index = record.map_edge_indices.long()
    data["map", "connects", "map"].edge_type = record.map_edge_types
    data.scene_number = int(record.scene_number)
    data.dataset_id = -1 if record.dataset_id is None else int(record.dataset_id)
    data.position_offset = record.position_offset
    data.prediction_origin = record.prediction_origin
    data.prediction_end = record.prediction_end
    # PyG removes attributes assigned None. Keep identical keys across scenes
    # so datasets with and without ego metadata can share a batch. The mask
    # distinguishes missing metadata even if -1 is a valid source agent ID.
    data.ego_agent_id = -1 if record.ego_agent_id is None else record.ego_agent_id
    data.has_ego_agent_id = record.ego_agent_id is not None
    return data


def _pad_full_hetero_time_axes(record: HeteroData, *, horizon_frames: int) -> HeteroData:
    # Copy attribute stores before replacing padded fields. PyG concatenates
    # tensors into independent batch storage in Batch.from_data_list().
    padded = shallow_copy(record)
    padded["agent"].features = _pad_along_dim(
        record["agent"].features,
        target=horizon_frames,
        dim=1,
    )
    padded["agent"].valid_mask = _pad_along_dim(
        record["agent"].valid_mask,
        target=horizon_frames,
        dim=1,
    )
    return padded


def _pad_forecast_hetero_time_axes(
    record: HeteroData,
    *,
    history_frames: int,
    future_frames: int,
) -> HeteroData:
    padded = shallow_copy(record)
    padded["agent"].history_features = _pad_left_along_dim(
        record["agent"].history_features,
        target=history_frames,
        dim=1,
    )
    padded["agent"].history_mask = _pad_left_along_dim(
        record["agent"].history_mask,
        target=history_frames,
        dim=1,
    )
    padded["agent"].future_features = _pad_along_dim(
        record["agent"].future_features,
        target=future_frames,
        dim=1,
    )
    padded["agent"].future_mask = _pad_along_dim(
        record["agent"].future_mask,
        target=future_frames,
        dim=1,
    )
    return padded


def _pad_along_dim(tensor: torch.Tensor, *, target: int, dim: int) -> torch.Tensor:
    current = int(tensor.size(dim))
    if current >= target:
        return tensor

    shape = list(tensor.shape)
    shape[dim] = target
    padded = tensor.new_zeros(shape)

    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, current)
    padded[tuple(slices)] = tensor
    return padded


def _pad_left_along_dim(tensor: torch.Tensor, *, target: int, dim: int) -> torch.Tensor:
    current = int(tensor.size(dim))
    if current >= target:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = target
    padded = tensor.new_zeros(shape)
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(target - current, target)
    padded[tuple(slices)] = tensor
    return padded
