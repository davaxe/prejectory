"""Loader implementations for the ETH/UCY pedestrian datasets."""

from __future__ import annotations

from functools import cached_property
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import override

from prejectory.config.models import ReadAll, TimeBlockAssign
from prejectory.core.categories import AgentCategory, DatasetSplit
from prejectory.core.errors import ConfigurationError
from prejectory.core.scene import POSITIONS_ONLY
from prejectory.datasets.eth_ucy.sequence import (
    DOMAIN_RECORDINGS,
    discover_recordings,
    read_recording,
)
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import (
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.config.models import DatasetConfig
    from prejectory.core.scene import TrajectorySchema


class EthUcyOptions(LoaderOptionsModel):
    """Select benchmark folds or reconstructed pedestrian domains."""

    input_mode: Literal["benchmark", "sequence"] = "benchmark"

    @override
    def validate_config(self, config: DatasetConfig) -> None:
        """Reject split settings that would invalidate sequence isolation."""
        if self.input_mode == "benchmark":
            if config.assign.strategy in {"time", "shuffled-time"}:
                msg = "ETH/UCY temporal assignment requires input_mode='sequence'."
                raise ConfigurationError(msg)
            return
        if not isinstance(config.read, ReadAll):
            msg = "ETH/UCY sequence mode requires read.strategy='all'; native folds are ignored."
            raise ConfigurationError(msg)
        if not isinstance(config.assign, TimeBlockAssign):
            msg = "ETH/UCY sequence mode requires assign.strategy='time' with explicit ratios."
            raise ConfigurationError(msg)
        if config.scenes.window is None or config.scenes.window.policy != "strict":
            msg = "ETH/UCY sequence mode requires strict scene windowing."
            raise ConfigurationError(msg)
        if not isclose(config.scenes.sample_time, 0.4, rel_tol=0, abs_tol=1e-12):
            msg = "ETH/UCY sequence input has sample_time=0.4; use scenes.resample to change rate."
            raise ConfigurationError(msg)


class EthUcyLoader(SceneLoader[Path | tuple[Path, ...], EthUcyOptions]):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    @cached_property
    def _recordings(self) -> dict[str, tuple[Path, ...]]:
        return discover_recordings(self.root, self.request.dataset or "")

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path | tuple[Path, ...]]]:
        if self.loader_options.input_mode == "benchmark":
            yield from super().iter_sources()
        else:
            for name, files in self._recordings.items():
                yield DatasetSource(identifier=name, payload=files)

    @override
    def count_sources(self) -> int | None:
        if self.loader_options.input_mode == "sequence":
            return len(self._recordings)
        return super().count_sources()

    @override
    def iter_sources_for(
        self, split: DatasetSplit
    ) -> Iterable[DatasetSource[Path | tuple[Path, ...]]]:
        if self.loader_options.input_mode == "sequence":
            msg = "Native fold selection is unavailable in ETH/UCY sequence mode."
            raise ConfigurationError(msg)
        for data_file in (self.root / split.value).rglob("*.txt"):
            yield DatasetSource(
                identifier=self._source_identifier(data_file=data_file, split=split),
                payload=data_file,
            )

    @override
    def load_source(
        self, source: DatasetSource[Path | tuple[Path, ...]]
    ) -> Iterable[LoadedSourceFrame]:
        if isinstance(source.payload, tuple):
            members = DOMAIN_RECORDINGS[self.request.dataset or ""]
            frame = read_recording(
                source.payload,
                recording_index=members.index(str(source.identifier)),
                recording_count=len(members),
            )
            yield LoadedSourceFrame(
                frame.lazy().with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN)),
            )
            return
        yield LoadedSourceFrame(
            pl.scan_csv(
                source.payload,
                has_header=False,
                separator="\t",
                new_columns=["frame", "id", "x", "y"],
                schema_overrides={
                    "frame": pl.Float64,
                    "id": pl.Float64,
                    "x": pl.Float64,
                    "y": pl.Float64,
                },
            ).with_columns(
                ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
                pl.col("id").cast(pl.Int32),
                agent_category=pl.lit(AgentCategory.PEDESTRIAN),
            ),
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        return sum(1 for _ in (self.root / split.value).rglob("*.txt"))

    def _source_identifier(self, *, data_file: Path, split: DatasetSplit) -> str:
        direct_split_dir = self.root / split.value
        if data_file.parent == direct_split_dir:
            return data_file.name
        return data_file.relative_to(self.root).as_posix()
