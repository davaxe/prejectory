"""Torch dataset adapters built on top of framework-neutral scene readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prejectory.core.errors import MissingPredictionBoundsError
from prejectory.core.optional import raise_missing_optional_dependency

try:
    from typing import override

    import torch
    from torch.utils.data import Dataset, IterableDataset
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(
        error,
        feature="The Torch scene dataset adapter",
        extra="torch",
    )


from prejectory.io.base import DatasetReader, IterableDatasetReader
from prejectory.io.records import PredictionBounds, SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True)
class TorchSceneRecord:
    """Torch-backed equivalent of full-horizon `SceneRecord`."""

    scene_number: int
    """Scene identifier within the exported dataset."""
    dataset_id: int | None
    """Dataset id associated with this scene, if known."""
    prediction_origin: int | None
    """First prediction frame associated with this scene, if known."""
    prediction_end: int | None
    """Exclusive prediction endpoint associated with this scene, if known."""
    position_offset: torch.Tensor
    """Global 2D translation offset with shape `(2,)`."""
    agent_ids: torch.Tensor
    """Source agent identifiers with shape `(N,)`."""
    agent_types: torch.Tensor
    """Integer-encoded agent types with shape `(N,)`."""
    features: torch.Tensor
    """Full-horizon agent features with shape `(N, T, F)`."""
    valid_mask: torch.Tensor
    """Validity mask for `features`, shape `(N, T)`."""
    screened_agent_mask: torch.Tensor
    """Mask of agents that passed screening, shape `(N,)`."""
    map_node_positions: torch.Tensor
    """2D map node coordinates with shape `(M, 2)`."""
    map_edge_indices: torch.Tensor
    """Directed map connectivity with shape `(2, E)`."""
    map_node_types: torch.Tensor
    """Integer-encoded map node types with shape `(M,)`."""
    map_edge_types: torch.Tensor
    """Integer-encoded map edge types with shape `(E,)`."""
    ego_agent_id: int | None
    """Optional source identifier of the ego agent."""

    def forecast(self, bounds: PredictionBounds | None = None) -> TorchForecastRecord:
        """Return history/future views sharing storage with this record.

        Bounds use stored (post-resampling) frames. Omitting them uses the
        record's prediction task; task-free records need explicit bounds.
        """
        if bounds is None:
            if self.prediction_origin is None or self.prediction_end is None:
                raise MissingPredictionBoundsError(self.scene_number)
            bounds = PredictionBounds(self.prediction_origin, self.prediction_end)
        bounds.validate(horizon_frames=int(self.features.size(1)))

        return TorchForecastRecord(
            scene_number=self.scene_number,
            dataset_id=self.dataset_id,
            prediction_origin=bounds.prediction_origin,
            prediction_end=bounds.prediction_end,
            position_offset=self.position_offset,
            agent_ids=self.agent_ids,
            agent_types=self.agent_types,
            screened_agent_mask=self.screened_agent_mask,
            history_features=self.features[:, : bounds.prediction_origin],
            history_mask=self.valid_mask[:, : bounds.prediction_origin],
            future_features=self.features[:, bounds.prediction_origin : bounds.prediction_end],
            future_mask=self.valid_mask[:, bounds.prediction_origin : bounds.prediction_end],
            map_node_positions=self.map_node_positions,
            map_edge_indices=self.map_edge_indices,
            map_node_types=self.map_node_types,
            map_edge_types=self.map_edge_types,
            ego_agent_id=self.ego_agent_id,
        )


@dataclass(slots=True)
class TorchForecastRecord:
    """Torch-backed equivalent of a split scene record."""

    scene_number: int
    dataset_id: int | None
    prediction_origin: int
    prediction_end: int
    position_offset: torch.Tensor
    agent_ids: torch.Tensor
    agent_types: torch.Tensor
    screened_agent_mask: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    future_features: torch.Tensor
    future_mask: torch.Tensor
    map_node_positions: torch.Tensor
    map_edge_indices: torch.Tensor
    map_node_types: torch.Tensor
    map_edge_types: torch.Tensor
    ego_agent_id: int | None


class TorchSceneDataset[ReaderT: DatasetReader[SceneRecord]](Dataset[TorchSceneRecord]):
    """Map-style Torch dataset over any `DatasetReader[SceneRecord]`."""

    def __init__(self, reader: ReaderT, *, copy: bool = True) -> None:
        super().__init__()
        self.reader: ReaderT = reader
        self._copy: bool = copy

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)

    @override
    def __getitem__(self, index: int) -> TorchSceneRecord:
        """Return one full-horizon scene record converted to Torch tensors."""
        return to_torch_scene_record(self.reader[index], copy=self._copy)


class IterableTorchSceneDataset[IterableReaderT: IterableDatasetReader[SceneRecord]](
    IterableDataset[TorchSceneRecord]
):
    """Iterable Torch dataset wrapper over any `DatasetReader[SceneRecord]`."""

    def __init__(self, reader: IterableReaderT, *, copy: bool = True) -> None:
        super().__init__()
        self.reader: IterableReaderT = reader
        self._copy: bool = copy

    @override
    def __iter__(self) -> Iterator[TorchSceneRecord]:
        """Iterate over full-horizon scene records converted to Torch tensors."""
        for record in self.reader:
            yield to_torch_scene_record(record, copy=self._copy)

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)


class TorchForecastDataset[ReaderT: DatasetReader[SceneRecord]](Dataset[TorchForecastRecord]):
    """Map-style Torch forecast view over full-horizon scene records."""

    def __init__(
        self,
        reader: ReaderT,
        *,
        bounds: PredictionBounds | None = None,
        copy: bool = True,
    ) -> None:
        super().__init__()
        self.dataset: TorchSceneDataset[ReaderT] = TorchSceneDataset(reader, copy=copy)
        self.bounds: PredictionBounds | None = bounds

    def __len__(self) -> int:
        """Report the number of records exposed by the wrapped reader."""
        return len(self.dataset)

    @override
    def __getitem__(self, index: int) -> TorchForecastRecord:
        return self.dataset[index].forecast(self.bounds)


class IterableTorchForecastDataset[IterableReaderT: IterableDatasetReader[SceneRecord]](
    IterableDataset[TorchForecastRecord],
):
    """Iterable Torch forecast view over full-horizon scene records."""

    def __init__(
        self,
        reader: IterableReaderT,
        *,
        bounds: PredictionBounds | None = None,
        copy: bool = True,
    ) -> None:
        super().__init__()
        self.dataset: IterableTorchSceneDataset[IterableReaderT] = IterableTorchSceneDataset(
            reader,
            copy=copy,
        )
        self.bounds: PredictionBounds | None = bounds

    @override
    def __iter__(self) -> Iterator[TorchForecastRecord]:
        for record in self.dataset:
            yield record.forecast(self.bounds)

    def __len__(self) -> int:
        """Report the number of records exposed by the wrapped reader."""
        return len(self.dataset)


def to_torch_scene_record(record: SceneRecord, *, copy: bool = True) -> TorchSceneRecord:
    """Convert a framework-neutral scene record into Torch tensors."""
    # Some readers can expose read-only NumPy views; copy once so Torch receives
    # writable, stable buffers without emitting warnings.
    return TorchSceneRecord(
        scene_number=record.scene_number,
        dataset_id=record.dataset_id,
        prediction_origin=record.prediction_origin,
        prediction_end=record.prediction_end,
        position_offset=torch.asarray(record.position_offset, copy=copy),
        agent_ids=torch.asarray(record.agent_ids, copy=copy),
        agent_types=torch.asarray(record.agent_types, copy=copy),
        screened_agent_mask=torch.asarray(record.screened_agent_mask, copy=copy),
        features=torch.asarray(record.features, copy=copy),
        valid_mask=torch.asarray(record.valid_mask, copy=copy),
        map_node_positions=torch.asarray(record.map_node_positions, copy=copy),
        map_edge_indices=torch.asarray(record.map_edge_indices, copy=copy),
        map_node_types=torch.asarray(record.map_node_types, copy=copy),
        map_edge_types=torch.asarray(record.map_edge_types, copy=copy),
        ego_agent_id=record.ego_agent_id,
    )
