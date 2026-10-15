"""Base loader implementation for thin dataset-ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Self, cast

from typing_extensions import TypeVar

from prejectory.core.typing import SourceT
from prejectory.processing.loading.models import LoaderOptionsModel, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.config.models import MapConfig, ScenesConfig, ScreeningConfig
    from prejectory.core.categories import DatasetSplit
    from prejectory.core.scene import TrajectorySchema
    from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame
    from prejectory.processing.maps import MapProvider
    from prejectory.processing.models import LoaderPlan, ReadSelection


_LoaderOptionsT = TypeVar("_LoaderOptionsT", bound=LoaderOptionsModel, default=NoLoaderOptions)


class SceneLoader(ABC, Generic[SourceT, _LoaderOptionsT]):
    """Base class for turning raw dataset sources into canonical trajectory frames.

    Parameters
    ----------
    data_root : Path or str
        The root directory of the dataset, which may be used for DatasetSource
        discovery or as a base path for DatasetSource data.
    request : LoaderPlan
        The full loader request, which may be used to configure loading behavior
        and is retained for potential use by subclasses.
    map_provider : MapProvider or None, optional
        Optional map provider prepared for the run and shared across sources.

    Notes
    -----
    - `SourceT` is the type of the raw DatasetSource data used by this loader, such as
      a file path or database query.
    - `_LoaderOptionsT` is the type of the dataset-specific options for this
      loader, which must be a subclass of `LoaderOptionsModel` and defaults to
      `NoLoaderOptions` for loaders without dataset-specific options.
    """

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> None:
        self.root: Path = Path(data_root)
        self.request: LoaderPlan = request
        self.scenes_config: ScenesConfig = request.scenes
        self.screening_config: ScreeningConfig | None = request.screening
        self.read_config: ReadSelection = request.read
        self.map_config: MapConfig | None = request.map
        self.map_provider: MapProvider | None = map_provider
        self.loader_options: _LoaderOptionsT = cast("_LoaderOptionsT", request.loader_options)

    def __init_subclass__(cls) -> None:
        """Ensure subclasses implement required DatasetSource enumeration methods."""
        if cls is SceneLoader:
            return

        if (
            cls.iter_sources_for is SceneLoader.iter_sources_for
            and cls.iter_sources is SceneLoader.iter_sources
        ):
            msg = f"{cls.__name__} must implement either iter_sources_for() or iter_sources()"
            raise TypeError(msg)

    @classmethod
    def from_loader_request(
        cls,
        data_root: Path | str,
        request: LoaderPlan,
        map_provider: MapProvider | None = None,
    ) -> Self:
        """Construct a concrete loader instance from the unified request interface.

        Parameters
        ----------
        data_root : Path or str
            Root directory or base path for the dataset.
        request : LoaderPlan
            Full loader request containing all generic and dataset-specific
            options.
        map_provider : MapProvider or None, optional
            Optional map provider prepared for the run and injected into the
            loader instance.

        Returns
        -------
        Self
            A concrete instance of the loader class.

        Notes
        -----
        This method exists to provide a consistent construction interface across
        all loaders, which is especially useful when loaders are selected
        dynamically at runtime.

        Subclasses may override this method if they need custom construction
        behavior, precomputation, validation, or dependency injection beyond the
        default initializer.
        """
        return cls(data_root=data_root, request=request, map_provider=map_provider)

    @classmethod
    @abstractmethod
    def native_trajectory_schema(cls) -> TrajectorySchema:
        """Return the native trajectory schema emitted by this loader.

        Returns
        -------
        TrajectorySchema
            The schema describing the raw trajectory fields produced before any
            downstream canonicalization, filtering, or transformation pipeline
            stages are applied.

        Notes
        -----
        This schema is used to derive column metadata and configure the
        processing pipeline appropriately. Concrete loaders should report the
        schema that matches what `load_source()` yields, not necessarily the
        final schema after pipeline execution.
        """

    @abstractmethod
    def load_source(self, source: DatasetSource[SourceT]) -> Iterable[LoadedSourceFrame]:
        """Load one dataset DatasetSource into one or more lazily consumable scene records.

        Parameters
        ----------
        source : DatasetSource[SourceT]
            A DatasetSource descriptor wrapping the dataset-specific raw DatasetSource object.
            Depending on the loader, this may represent a file, shard, query, or
            any other unit of dataset input.

        Returns
        -------
        Iterable[LoadedSourceFrame]
            An iterable of loaded DatasetSource records. Each record typically contains
            lazy frame data and may optionally include DatasetSource-level metadata such
            as map references or split annotations.

        Notes
        -----
        Implementations should treat this as the primary bridge between raw
        dataset storage and the canonical internal scene-loading pipeline.

        The iterable may be lazy. This is often preferable for large datasets,
        since it allows streaming scene materialization rather than requiring the
        entire DatasetSource to be loaded into memory at once.
        """

    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[SourceT]]:
        """Yield sources matching a specific selection request.

        !!! warning "Implement at least one DatasetSource enumeration method"
            Subclasses should implement either `iter_sources_for()` with native
            split support or `iter_sources()` for full enumeration. The default
            implementation of `iter_sources_for()` raises `NotImplementedError`
            to enforce this contract and provide a clear error message if
            neither method is implemented.

        Parameters
        ----------
        split : DatasetSplit
            The dataset split to select sources for.

        Returns
        -------
        Iterable[DatasetSource[SourceT]]
            An iterable of DatasetSource descriptors matching the selection.

        Raises
        ------
        NotImplementedError
            If the concrete loader does not implement DatasetSource enumeration.

        Notes
        -----
        `selection.native_split=None` represents all sources exposed by the
        loader. Native-split datasets may also support selection of one
        concrete native partition via `selection.native_split`.
        """  # ruff: ignore[docstring-extraneous-returns]
        _ = split
        msg = (
            f"{self.__class__.__name__} does not implement DatasetSource enumeration"
            " by split. Either implement iter_sources_for() with native split"
            " support or iter_sources() for full enumeration."
        )
        raise NotImplementedError(msg)

    def iter_sources(self) -> Iterable[DatasetSource[SourceT]]:
        """Yield all sources for the effective read scope.

        !!! warning "Implement at least one DatasetSource enumeration method"
            Subclasses should implement either `iter_sources_for()` with native
            split support or `iter_sources()` for full enumeration. The default
            implementation of `iter_sources_for()` raises `NotImplementedError`
            to enforce this contract and provide a clear error message if
            neither method is implemented.

        !!! note "Default implementation"
            The default implementation of `iter_sources()` checks for native split
            support and delegates to `iter_sources_for()` if native splits are
            defined. This allows loaders with native split support to only
            implement `iter_sources_for()` while still providing a working
            `iter_sources()` method for full enumeration. Loaders without native
            split support must implement `iter_sources()` directly.

        """
        native_splits = self.read_config.native_splits
        if native_splits is not None:
            for split in native_splits:
                for source in self.iter_sources_for(split):
                    yield source.with_source_split(split)
            return
        msg = (
            f"{self.__class__.__name__} does not implement iter_sources()"
            f"or `iter_sources_for()` with native split support"
        )
        raise NotImplementedError(msg)

    def count_sources_for(self, split: DatasetSplit) -> int | None:
        """Return the DatasetSource count for a selection when cheaply knowable.

        !!! info "Override when DatasetSource counts are cheaply knowable"
            The default implementation returns `None` to indicate that the count
            is unknown or expensive to compute. Loaders with native split
            support and cheaply knowable DatasetSource counts should override this
            method to enable accurate progress reporting and diagnostics.

            If the native splits is not defined, overrides should implement
            `count_sources()` to provide a total count.

        Returns
        -------
        int or None
            The number of sources matching the selection, or `None` if the value
            is unknown, expensive to compute, or inherently dynamic.
        """
        _ = split, self
        return None

    def count_sources(self) -> int | None:
        """Return the total number of sources for the effective read scope.

        Returns
        -------
        int or None
            The number of sources available to the loader, or `None` if the
            value is unknown, expensive to compute, or inherently dynamic.

        Notes
        -----
        This method is primarily useful for progress reporting, scheduling, and
        diagnostics. Implementations should only return an integer when doing so
        does not require materializing or exhaustively traversing the full
        DatasetSource iterator unless that cost is acceptable.
        """
        native_splits = self.read_config.native_splits
        if native_splits is None:
            return None
        total = 0
        for split in native_splits:
            count = self.count_sources_for(split)
            if count is None:
                return None
            total += count
        return total
