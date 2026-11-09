"""Map-graph builder for the AD4CHE dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, override

import cv2
import numpy as np
import numpy.typing as npt

from prejectory.core.categories import EdgeType
from prejectory.processing.maps import (
    FeatureMapBuilder,
    MapProvider,
    MapReference,
    PathFeature,
    apply_map_config,
    extract_based_on_scene,
    resolve_map_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from prejectory.config.models import MapConfig
    from prejectory.core.maps import MapGraph
    from prejectory.core.scene import Scene


PIXEL_TO_METER: Final[float] = 0.038


@dataclass(slots=True)
class AD4CHEMapProvider(MapProvider):
    root: Path
    config: MapConfig
    _configured_map_cache: dict[str, MapGraph] = field(default_factory=dict, init=False)

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = resolve_map_key(scene, reference)
        if key is None:
            return None

        map_graph = self._configured_map(str(key))
        return extract_based_on_scene(map_graph, scene, self.config.extraction)

    def _configured_map(self, map_key: str) -> MapGraph:
        cached = self._configured_map_cache.get(map_key)
        if cached is not None:
            return cached

        map_graph = AD4CHEMapBuilder(self.root / map_key).build(
            min_distance=self.config.min_distance,
            interpolation_distance=self.config.interpolation_distance,
        )
        configured = apply_map_config(map_graph, self.config)
        self._configured_map_cache[map_key] = configured
        return configured


class AD4CHEMapBuilder(FeatureMapBuilder):
    """Map builder for the AD4CHE dataset."""

    def __init__(
        self,
        map_image_path: Path,
        pixel_to_meter: float = PIXEL_TO_METER,
        spatial_ds: float = 4.0,
    ) -> None:
        if not map_image_path.is_file():
            msg = f"Map image file does not exist: {map_image_path}"
            raise ValueError(msg)

        self.map_image_path: Path = map_image_path
        self.pixel_to_meter: float = pixel_to_meter
        self.spatial_ds: float = spatial_ds

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        gray = cv2.imread(str(self.map_image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            msg = f"Failed to load image: {self.map_image_path}"
            raise ValueError(msg)

        gray = cv2.flip(gray, 0)
        h_pixels, _ = gray.shape
        h_meters = h_pixels * self.pixel_to_meter

        border_mask = _get_black_border_pixels(gray.astype(np.uint8))
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) <= 1:
                continue
            coords = np.asarray(cnt[:, 0, :], dtype=np.float64) * self.pixel_to_meter
            coords = _spatial_downsample_polyline(coords, d_min=self.spatial_ds)
            if len(coords) < 2:
                continue
            points = [(float(pt[0]), h_meters - float(pt[1])) for pt in coords]
            yield PathFeature(points=tuple(points), edge_types=EdgeType.LINE_THICK_DASHED)


def _get_black_border_pixels(gray_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    black_mask = (gray_image == 0).astype(np.uint8)
    white_mask = (gray_image > 0).astype(np.uint8)
    dilated_white = cv2.dilate(white_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(black_mask == 1, dilated_white == 1).astype(np.uint8) * 255


def _spatial_downsample_polyline(
    polyline: npt.NDArray[np.float64],
    d_min: float = 1,
) -> npt.NDArray[np.float64]:
    if len(polyline) < 2:
        return polyline

    filtered = [polyline[0]]
    for pt in polyline[1:]:
        if np.linalg.norm(pt - filtered[-1]) >= d_min:
            filtered.append(pt)

    return np.array(filtered)
