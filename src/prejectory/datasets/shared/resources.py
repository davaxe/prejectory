"""Shared helpers for dataset map-provider factories."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from prejectory.config.models import MapConfig, ScenesConfig
from prejectory.core.maps import MapGraph
from prejectory.processing.maps import MapProvider, SharedMapProvider, apply_map_config, extract_fn

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.shared_memory import SharedMemory


type MapBuilder = Callable[[Path, MapConfig], MapGraph]
type NamedPathsFactory = Callable[[Path], Iterable[tuple[str | None, Path]]]
type SinglePathFactory = Callable[[Path], Path]
type MapProviderFactory = Callable[
    [Path, ScenesConfig, MapConfig | None],
    AbstractContextManager[MapProvider | None],
]


@contextmanager
def open_named_shared_map_resources(
    *,
    map_config: MapConfig | None,
    named_paths: Iterable[tuple[str | None, Path]],
    build_map: MapBuilder,
) -> Generator[MapProvider | None]:
    """Open shared-memory map resources keyed by explicit names."""
    if map_config is None:
        yield None
        return

    handles: list[SharedMemory] = []
    mappings: dict[str | None, str] = {}
    for key, path in named_paths:
        handle = apply_map_config(build_map(path, map_config), map_config).to_shared()
        handles.append(handle)
        mappings[key] = handle.name

    try:
        yield SharedMapProvider(shared_names=mappings, extractor=extract_fn(map_config.extraction))

    finally:
        for handle in handles:
            handle.close()
            handle.unlink()


@contextmanager
def open_single_shared_map_resource(
    *,
    map_config: MapConfig | None,
    map_path: Path,
    build_map: MapBuilder,
) -> Generator[MapProvider | None]:
    """Open a single shared-memory map resource."""
    if map_config is None:
        yield None
        return

    handle = apply_map_config(build_map(map_path, map_config), map_config).to_shared()
    try:
        yield SharedMapProvider(
            shared_names=handle.name,
            extractor=extract_fn(map_config.extraction),
        )
    finally:
        handle.close()
        handle.unlink()


def named_shared_map_resources_factory(
    *,
    named_paths: NamedPathsFactory,
    build_map: MapBuilder,
) -> MapProviderFactory:
    """Return a registry map-provider factory for named shared maps."""

    @contextmanager
    def _factory(
        root: Path,
        scenes: ScenesConfig,
        map_config: MapConfig | None,
    ) -> Generator[MapProvider | None]:
        _ = scenes
        with open_named_shared_map_resources(
            map_config=map_config,
            named_paths=named_paths(root),
            build_map=build_map,
        ) as map_provider:
            yield map_provider

    return _factory


def single_shared_map_resource_factory(
    *,
    map_path: SinglePathFactory,
    build_map: MapBuilder,
) -> MapProviderFactory:
    """Return a registry map-provider factory for one shared map."""

    @contextmanager
    def _factory(
        root: Path,
        scenes: ScenesConfig,
        map_config: MapConfig | None,
    ) -> Generator[MapProvider | None]:
        _ = scenes
        with open_single_shared_map_resource(
            map_config=map_config,
            map_path=map_path(root),
            build_map=build_map,
        ) as map_provider:
            yield map_provider

    return _factory


def map_provider_resources_factory(
    *,
    create: Callable[[Path, MapConfig], MapProvider],
) -> MapProviderFactory:
    """Return a map-provider factory that installs a per-run `MapProvider`."""

    @contextmanager
    def _factory(
        root: Path,
        scenes: ScenesConfig,
        map_config: MapConfig | None,
    ) -> Generator[MapProvider | None]:
        _ = scenes

        if map_config is None:
            yield None
            return
        yield create(root, map_config)

    return _factory
