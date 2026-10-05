# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from typing_extensions import TypeAliasType, override

if TYPE_CHECKING:
    from collections.abc import Iterator

ResampleMethod: TypeAlias = Literal["linear", "cubic", "pchip"]

PatchDumpValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | tuple["PatchDumpValue", ...]
    | list["PatchDumpValue"]
    | dict[object, "PatchDumpValue"]
    | None
)


class ConfigBase(BaseModel):
    """Shared base class for all config models."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @override
    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        for name, field in super().__repr_args__():
            if field is not None:
                yield name, field


class ResolvedConfig(ConfigBase):
    """Full configuration with all fields required and defaults applied."""


class Clear(ConfigBase):
    """Patch operation that clears an optional inherited config block."""

    op: Literal["clear"] = Field(default="clear", init=False)


ResolvedConfigT = TypeVar("ResolvedConfigT", bound=ConfigBase)


class ConfigPatch(ConfigBase, Generic[ResolvedConfigT]):
    """Base class for patch models applied to resolved config models.

    Patch semantics are based on fields explicitly supplied to the patch model,
    not on whether their values are `None`. This means nullable resolved fields
    can be intentionally set to `None` by providing the field in the patch.
    """

    full_config_type: type[ResolvedConfigT]

    def merge_into(self, target: ResolvedConfigT | None) -> ResolvedConfigT:
        """Apply this patch to a resolved target config.

        If `target` is `None`, the patch is applied to the defaults of the
        target config type. Required fields that remain missing are rejected by
        final validation.
        """
        if target is None:
            base = {
                name: field.get_default(call_default_factory=True)
                for name, field in self.full_config_type.model_fields.items()
                if not field.is_required()
            }
        else:
            base = target.model_dump(mode="python")

        patch = patch_model_dump(self)
        merged = deep_merge(base, patch)
        return self.full_config_type.model_validate(merged)


def patch_model_dump(model: BaseModel) -> dict[str, object]:
    """Dump only fields explicitly provided to a patch model.

    Unlike `model_dump(exclude_none=True)`, this preserves explicit `None`
    values. That is required for nullable resolved fields such as
    `map.min_distance` or `runtime.chunksize`.
    """
    data: dict[str, object] = {}
    for name in model.model_fields_set:
        value = getattr(model, name)
        data[name] = _dump_patch_value(value)
    return data


def _dump_patch_value(value: object) -> PatchDumpValue:
    if isinstance(value, ConfigPatch):
        return cast("PatchDumpValue", patch_model_dump(value))
    if isinstance(value, BaseModel):
        return cast("PatchDumpValue", value.model_dump(mode="python"))
    if isinstance(value, tuple):
        return tuple(_dump_patch_value(item) for item in value)
    if isinstance(value, list):
        return [_dump_patch_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _dump_patch_value(item) for key, item in value.items()}

    return cast("PatchDumpValue", copy.deepcopy(value))


def deep_merge(
    base: MutableMapping[str, object],
    patch: Mapping[str, object],
) -> MutableMapping[str, object]:
    """Recursively merge two mappings.

    Nested mappings are merged. All other values replace the target value.
    Discriminated unions are normally represented as mappings, but patch models
    should provide complete replacement values for union fields.
    """
    for key, patch_value in patch.items():
        base_value = base.get(key)
        if isinstance(base_value, MutableMapping) and isinstance(patch_value, Mapping):
            _ = deep_merge(
                cast("MutableMapping[str, object]", base_value),
                cast("Mapping[str, object]", patch_value),
            )
        else:
            base[key] = copy.deepcopy(patch_value)
    return base


TargetT = TypeVar("TargetT", bound=ConfigBase)


def apply_optional(
    patch: ConfigPatch[TargetT] | Clear | None,
    target: TargetT | None,
) -> TargetT | None:
    """Apply a patch to an optional config block."""
    if patch is None:
        return target
    if isinstance(patch, Clear):
        return None
    return patch.merge_into(target)


ValueT = TypeVar("ValueT")

_MAPPING_PATCH_RESERVED_KEYS = frozenset({"mode", "remove", "values"})


class MappingPatch(ConfigBase, Generic[ValueT]):
    """Patch operation for named mapping fields.

    `replace` discards the inherited mapping before applying values.
    `extend` starts from the inherited mapping. `remove` is section-local.
    """

    mode: Literal["replace", "extend"] = "extend"
    remove: tuple[str, ...] = ()
    values: dict[str, ValueT] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_compact_mapping_patch(cls, data: Any) -> Any:  # ruff: ignore[any-type]
        """Normalize input to allow compact syntax for simple mapping patches."""
        if not isinstance(data, Mapping):
            return data
        normalized: dict[str, Any] = {}
        if "mode" in data:
            normalized["mode"] = data["mode"]
        if "remove" in data:
            normalized["remove"] = data["remove"]

        values: dict[str, Any] = {}

        explicit_values = data.get("values")
        if explicit_values is not None:
            if not isinstance(explicit_values, Mapping):
                msg = "`values` must be a mapping when provided."
                raise TypeError(msg)
            values.update(explicit_values)

        for key, value in data.items():
            if key in _MAPPING_PATCH_RESERVED_KEYS:
                continue
            if not isinstance(key, str):
                msg = "Compact mapping patch keys must be strings."
                raise TypeError(msg)
            values[key] = value
        normalized["values"] = values
        return normalized

    def merge_into(self, target: Mapping[str, ValueT] | None) -> dict[str, ValueT]:
        """Apply this patch to an optional target mapping."""
        result: dict[str, ValueT] = {} if self.mode == "replace" else dict(target or {})
        for key in self.remove:
            _ = result.pop(key, None)
        result.update(self.values)
        return result


class DictPatch(MappingPatch[object]):
    """Patch operation for arbitrary string-keyed dictionaries."""


def _clear_shorthand(value: object) -> object:
    if value == "clear":
        return {"op": "clear"}
    return value


T = TypeVar("T")

Clearable = TypeAliasType(
    "Clearable",
    Annotated[T | Clear | None, BeforeValidator(_clear_shorthand)],
    type_params=(T,),
)
