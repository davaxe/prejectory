"""Loader implementation for the OpenDD dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.categories import AgentCategory
from prejectory.core.scene import POSITIONS_ONLY
from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import DatasetSource, LoadedSourceFrame, NoLoaderOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prejectory.core.scene import TrajectorySchema


def _table_query(table_name: str) -> str:
    return f"""
    SELECT
        OBJID as id,
        TIMESTAMP,
        UTM_X as x,
        UTM_Y as y,
        CLASS
    FROM {table_name}
    """  # ruff: ignore[hardcoded-sql-expression]


class OpenDDLoader(SceneLoader[tuple[Path, str], NoLoaderOptions]):
    """Loader for OpenDD data split across multiple SQLite databases."""

    def _db_paths(self) -> Iterable[Path]:
        for db_file in self.root.rglob("trajectories_*_v3.sqlite"):
            level = len(db_file.relative_to(self.root).parts)
            if level <= 3:
                yield db_file

    @override
    def iter_sources(self) -> Iterable[DatasetSource[tuple[Path, str]]]:
        for db_path in self._db_paths():
            for table_name in _list_table_names(db_path):
                identifier = f"{db_path.stem}:{table_name}"
                yield DatasetSource(
                    identifier=identifier,
                    payload=(db_path, table_name),
                    map_key=str(_map_sqlite_path(db_path.parent)),
                )

    @override
    def count_sources(self) -> int | None:
        return sum(_count_tables(db_path) for db_path in self._db_paths())

    @override
    def load_source(self, source: DatasetSource[tuple[Path, str]]) -> Iterable[LoadedSourceFrame]:
        db_path, table_name = source.payload
        with sqlite3.connect(db_path) as connection:
            yield LoadedSourceFrame(
                pl
                .read_database(_table_query(table_name), connection)
                .lazy()
                .with_columns(
                    ((pl.col("TIMESTAMP") * 1000).round(4).rank(method="dense") - 1)
                    .cast(pl.Int64)
                    .alias("frame"),
                    pl
                    .col("CLASS")
                    .replace_strict(
                        {
                            "Car": AgentCategory.CAR,
                            "Medium Vehicle": AgentCategory.CAR,
                            "Heavy Vehicle": AgentCategory.TRUCK,
                            "Trailer": AgentCategory.TRUCK,
                            "Bus": AgentCategory.BUS,
                            "Motorcycle": AgentCategory.MOTORCYCLE,
                            "Pedestrian": AgentCategory.PEDESTRIAN,
                            "Bicycle": AgentCategory.BICYCLE,
                        },
                        default=AgentCategory.UNKNOWN,
                    )
                    .alias("agent_category"),
                )
                .drop("CLASS", "TIMESTAMP"),
            )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY


def _list_table_names(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        return [name for (name,) in cursor]


def _count_tables(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table';",
        ).fetchone()
    return row[0] if row is not None else 0


def _map_sqlite_path(recording_dir: Path) -> Path:
    return recording_dir / f"map_{recording_dir.name}" / f"map_{recording_dir.name}.sqlite"
