"""Scene containers and schema-conversion helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, override

import polars as pl

from prejectory.core.errors import TrajectorySchemaError
from prejectory.core.maps import MapGraph
from prejectory.core.scene.derivations import (
    ConversionContext,
    apply_derivation_plan,
    plan_derivations,
)
from prejectory.core.scene.schema import CANONICAL, TrajectoryField, TrajectorySchema

if TYPE_CHECKING:
    from prejectory.core.categories import DatasetSplit


type MapResolver = Callable[[Scene], MapGraph | None]
"""Callable signature used to materialize a scene's map graph lazily.

Resolvers are attached to [`Scene`][prejectory.core.scene.Scene] instances so
map loading can be deferred until a consumer explicitly requests it through
`Scene.resolve_map()`.
"""


@dataclass(slots=True, frozen=True)
class Scene:
    """Canonical scene wrapper plus schema metadata."""

    frame: pl.DataFrame
    """DataFrame containing the scene data."""
    scene_number: int
    """Unique scene number assigned during processing."""
    horizon_frames: int
    """Number of frames in the scene horizon."""
    schema: TrajectorySchema
    """Schema describing which fields this scene currently provides."""
    sample_time: float | None = None
    """Time interval between frames in seconds."""
    map_key: str | None = None
    """Stable map identifier for the scene, if one exists."""
    map_resolver: MapResolver | None = field(default=None, compare=False, repr=False)
    """Resolver attached by the loader to materialize the scene map on demand."""
    passed_agent_ids: frozenset[int] | None = None
    """Optional set of agents that passed screening and are retained for outputs."""
    split_assignment: DatasetSplit | None = None
    """Split assignment for this scene (train/val/test)."""
    dataset: str | None = None
    """Dataset label associated with this scene, if known."""
    ego_agent_id: int | None = None
    """Optional ID of the ego vehicle in this scene, if applicable."""

    @classmethod
    def create(
        cls,
        frame: pl.DataFrame,
        scene_number: int,
        *,
        horizon_frames: int,
        schema: TrajectorySchema,
        sample_time: float | None = None,
        map_key: str | None = None,
        map_resolver: MapResolver | None = None,
        split_assignment: DatasetSplit | None = None,
        dataset: str | None = None,
        passed_agent_ids: frozenset[int] | None = None,
        cast_schema: bool = True,
        ego_agent_id: int | None = None,
    ) -> Scene:
        """Create a scene with optional schema casting.

        Parameters reflect the fields of `Scene` but with an additional
        `cast_schema` flag that controls whether the input frame should be cast
        to match the expected physical schema of the provided trajectory schema.
        This allows loaders to provide frames that may have compatible but not
        exactly matching schemas, while ensuring that the resulting `Scene`
        instance adheres to the expected schema for downstream processing.

        Parameters
        ----------
        frame : pl.DataFrame
            DataFrame containing the scene data. Columns should correspond to the
            fields defined in the provided `schema`.
        scene_number : int
            Unique scene number assigned during processing.
        horizon_frames : int
            Number of frames included in the scene horizon.
        schema : TrajectorySchema
            Schema describing which fields the scene currently provides. The
            input frame is expected to have columns matching the physical fields
            defined by this schema.
        sample_time : float or None, optional
            Time interval between frames in seconds.
        map_key : MapKey, optional
            Stable map identifier for the scene, if one exists.
        map_resolver : MapResolver or None, optional
            Resolver attached by the loader to materialize the scene map on
            demand.
        split_assignment : DatasetSplit or None, optional
            Split assignment for this scene (train/val/test).
        dataset : str or None, optional
            Dataset label associated with this scene, if known.
        passed_agent_ids : frozenset of int or None, optional
            Optional set of agents that passed screening and should be explictly
            added to the scene metadata.
        cast_schema : bool, optional
            Whether to cast the input frame to match the expected physical
            schema or to assume it already matches. Default is `True`.
        ego_agent_id : int or None, optional
            Optional ID of the ego vehicle in this scene, if applicable.

        """
        if cast_schema:
            frame = _cast_to_schema(frame, schema.physical)

        return cls(
            frame=frame,
            scene_number=scene_number,
            horizon_frames=horizon_frames,
            schema=schema,
            sample_time=sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            passed_agent_ids=passed_agent_ids,
            split_assignment=split_assignment,
            dataset=dataset,
            ego_agent_id=ego_agent_id,
        )

    def __post_init__(self) -> None:
        """Cast and validate the trajectory schema."""
        if self.horizon_frames <= 0:
            msg = f"`horizon_frames` must be positive, but got {self.horizon_frames}."
            raise ValueError(msg)
        if not _matches_physical_schema(self.frame.schema, self.schema.physical):
            msg = _get_schema_mismatch_message(
                actual=self.frame.schema,
                expected=self.schema.physical,
                schema_name=self.schema.name,
            )
            raise TrajectorySchemaError(msg)
        _validate_scene_frame(self.frame, horizon_frames=self.horizon_frames)

    def resolve_map(self) -> MapGraph | None:
        """Materialize the map graph associated with this scene, if available."""
        if self.map_resolver is None:
            return None
        return self.map_resolver(self)

    def has_map(self) -> bool:
        """Return whether this scene has a lazy map resolver attached."""
        return self.map_key is not None or self.map_resolver is not None

    def as_schema(self, schema: TrajectorySchema = CANONICAL) -> Scene:
        """Return a copy converted to the requested trajectory schema."""
        if self.schema == schema and _matches_physical_schema(self.frame.schema, schema.physical):
            return self

        return convert_scene(self, schema)

    def with_split_assignment(self, split_assignment: DatasetSplit | None) -> Scene:
        """Return a copy with the given split assignment."""
        return replace(self, split_assignment=split_assignment)

    @override
    def __repr__(self) -> str:
        """Return a compact representation with metadata and DataFrame shape."""
        rows, cols = self.frame.shape
        return (
            "Scene("
            f"scene_number={self.scene_number}, "
            f"horizon_frames={self.horizon_frames}, "
            f"schema={self.schema.name}, "
            f"dataset={self.dataset!r}, "
            f"map_key={self.map_key!r}, "
            f"frame=DataFrame({rows} rows x {cols} cols)"
            ")"
        )


def convert_scene(scene: Scene, target: TrajectorySchema = CANONICAL) -> Scene:
    """Convert a scene to the requested schema."""
    return replace(
        scene,
        frame=_convert_frame(
            scene.frame,
            source=scene.schema,
            target=target,
            sample_time=scene.sample_time,
        ),
        schema=target,
    )


def derived_trajectory_fields(
    source: TrajectorySchema,
    target: TrajectorySchema,
    *,
    sample_time: float | None,
) -> tuple[TrajectoryField, ...]:
    """Return target schema fields that would be materialized by conversion.

    The result is ordered according to the target schema. A `TrajectorySchemaError` is
    raised when the requested conversion cannot be planned with the available
    source fields and sampling metadata.

    """
    missing_fields = target.fields & ~source.fields
    if not missing_fields:
        return ()

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.fields, target.fields, context)
    if plan is None:
        msg = f"No derivation plan found to convert from schema {source.name} to {target.name} "
        raise TrajectorySchemaError(msg)

    return tuple(field for field in target.ordered_fields() if (missing_fields & field) == field)


def _convert_frame(
    data: pl.DataFrame,
    *,
    source: TrajectorySchema,
    target: TrajectorySchema,
    sample_time: float | None,
) -> pl.DataFrame:
    """Convert a data frame from one trajectory schema to another."""
    semantic = _to_semantic_frame(data, source).sort(["id", "frame"])
    semantic = _derive_missing_fields(
        semantic,
        source=source,
        target=target,
        sample_time=sample_time,
    )
    return semantic.select([pl.col(field).cast(dtype) for field, dtype in target.physical.items()])


def _to_semantic_frame(data: pl.DataFrame, schema: TrajectorySchema) -> pl.DataFrame:
    return data.select(schema.semantic_fields())


def _derive_missing_fields(
    data: pl.DataFrame,
    *,
    source: TrajectorySchema,
    target: TrajectorySchema,
    sample_time: float | None,
) -> pl.DataFrame:
    if (source.fields & target.fields) == target.fields:
        return data

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.fields, target.fields, context)
    if plan is None:
        msg = f"No derivation plan found to convert from schema {source.name} to {target.name} "
        raise TrajectorySchemaError(msg)

    output, output_fields = apply_derivation_plan(data, plan, context, source.fields)
    if (target.fields & output_fields) != target.fields:
        missing = ", ".join(field.to_str() for field in (target.fields & ~output_fields).fields())
        msg = f"Cannot materialize trajectory schema {target.name}; missing {missing}."
        raise TrajectorySchemaError(msg)
    return output


def _cast_to_schema(data: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Cast the columns of a DataFrame to match the provided schema.

    This does not remove extra columns or add missing columns; it only casts
    existing columns to the specified types.

    """
    if _matches_physical_schema(data.schema, schema):
        return data
    missing = [column for column in schema if column not in data.schema]
    if missing:
        msg = _get_schema_mismatch_message(actual=data.schema, expected=schema)
        raise TrajectorySchemaError(msg)
    casts = [pl.col(col).cast(dtype) for col, dtype in schema.items() if data.schema[col] != dtype]
    return data if not casts else data.with_columns(casts)


def _validate_scene_frame(data: pl.DataFrame, *, horizon_frames: int) -> None:
    if data.is_empty():
        msg = "Scene frame must contain at least one trajectory observation."
        raise ValueError(msg)

    required = ("frame", "id", "x", "y", "agent_category")
    null_counts = data.select([pl.col(column).is_null().any().alias(column) for column in required])
    if any(bool(value) for value in null_counts.row(0)):
        msg = "Scene frame contains null values in required trajectory columns."
        raise ValueError(msg)

    if not data.select(pl.col("x").is_finite().all() & pl.col("y").is_finite().all()).item():
        msg = "Scene positions must contain only finite values."
        raise ValueError(msg)

    duplicates = data.select(pl.struct("id", "frame").is_duplicated().any()).item()
    if duplicates:
        msg = "Scene frame contains duplicate observations for an agent and frame."
        raise ValueError(msg)

    frame_min, frame_max = data.select(
        pl.col("frame").min().alias("frame_min"),
        pl.col("frame").max().alias("frame_max"),
    ).row(0)
    if int(frame_max) - int(frame_min) >= horizon_frames:
        msg = (
            "Scene frame span exceeds the declared horizon: "
            f"frames {frame_min}..{frame_max}, horizon {horizon_frames}."
        )
        raise ValueError(msg)


def _matches_physical_schema(actual: pl.Schema, expected: pl.Schema) -> bool:
    return all(column in actual and actual[column] == dtype for column, dtype in expected.items())


def _get_schema_mismatch_message(
    actual: pl.Schema,
    expected: pl.Schema,
    schema_name: str | None = None,
) -> str:
    missing = [col for col in expected if col not in actual]
    mismatched = {
        col: f"expected {expected[col]}, got {actual[col]}"
        for col in expected
        if col in actual and actual[col] != expected[col]
    }

    errors: list[str] = []
    if missing:
        errors.append(f"Missing fields: {missing}")
    if mismatched:
        errors.append(f"Mismatched types: {mismatched}")

    if not errors:
        return ""

    prefix = f"Schema mismatch for {schema_name}" if schema_name else "Schema mismatch"
    return f"{prefix}: {'; '.join(errors)}"
