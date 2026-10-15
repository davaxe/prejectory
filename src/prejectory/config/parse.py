from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from prejectory.config.base import Clear, ConfigBase
from prejectory.config.models import (
    DatasetConfig,
    DatasetConfigPatch,
    DatasetConfigPatchBase,
    PredictionTaskConfig,
)
from prejectory.core.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from prejectory.datasets.registry import DatasetDescriptor


def parse_config(path: str | Path) -> ProjectConfig:
    """Parse configuration from a TOML file.

    !!! note "Completeness of the returned config"
        By design, this loader returns a validated but patch-style project configuration for
        specific datasets.

        See the
        [`ProjectConfig.resolve_dataset_config`][prejectory.config.ProjectConfig.resolve_dataset_config]
        method for applying dataset-specific overrides returned by this loader
        on top of a fully resolved `DatasetConfig` to get a complete
        configuration for a specific dataset.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    ProjectConfig
        The parsed configuration, validated and ready for dataset-specific resolution.
    """
    path = Path(path)
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Invalid TOML in config file '{path}': {exc}"
            raise ConfigurationError(msg) from exc
    return ProjectConfig.model_validate(data)


class DatasetConfigEntry(DatasetConfigPatchBase):
    """Dataset-local authored config."""

    uses: tuple[str, ...] | None = None
    task: PredictionTaskConfig | Clear | str | None = None
    """Named task, complete inline task, or `none` for task-free output."""

    @field_validator("task", mode="before")
    @classmethod
    def _normalize_no_task(cls, value: object) -> object:
        return {"op": "clear"} if value == "none" else value

    def to_dataset_config_patch(self) -> DatasetConfigPatch:
        """Return config without the uses field."""
        return DatasetConfigPatch(
            scenes=self.scenes,
            task=None if isinstance(self.task, str) else self.task,
            runtime=self.runtime,
            screening=self.screening,
            loader_options=self.loader_options,
            output=self.output,
            map=self.map,
            read=self.read,
            assign=self.assign,
        )


class ProjectConfig(ConfigBase):
    """Root config model for processing requests.

    This model is designed to be loaded from a TOML file with potentially
    incomplete dataset entries, which can be resolved to complete
    [`DatasetConfig`][prejectory.config.models.DatasetConfig]s using the
    `resolve` method.

    To load a `ProjectConfig` from a TOML file, use the
    [`parse_config`][prejectory.config.parse_config] function.

    """

    defaults: DatasetConfigEntry | None = Field(default=None)
    profiles: dict[str, DatasetConfigPatch] = Field(default_factory=dict)
    datasets: dict[str, DatasetConfigEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_named_task_scope(self) -> ProjectConfig:
        if self.defaults is not None and isinstance(self.defaults.task, str):
            msg = "Named tasks may only be selected in a dataset-specific config entry."
            raise ConfigurationError(msg)
        return self

    def task_selection_for(self, dataset: str) -> PredictionTaskConfig | Clear | str | None:
        """Return the last authored task selection in project merge order."""
        selection: PredictionTaskConfig | Clear | str | None = None
        for entry in (self.defaults, self.datasets.get(dataset)):
            if entry is None:
                continue
            for use in entry.uses or ():
                profile = self.profiles.get(use)
                if profile is not None and profile.task is not None:
                    selection = profile.task
            if entry.task is not None:
                selection = entry.task
        return selection

    def resolve_dataset_config(self, dataset: str | DatasetDescriptor) -> DatasetConfig:
        """Apply project configuration to a dataset name or descriptor's defaults."""
        from prejectory.datasets.registry import get_dataset  # ruff: ignore[import-outside-top-level]

        descriptor = get_dataset(dataset) if isinstance(dataset, str) else dataset
        return self._resolve_config(
            descriptor.name,
            descriptor.default_config,
            named_tasks=descriptor.tasks,
            default_task=descriptor.default_task,
        )

    def _resolve_config(
        self,
        dataset: str,
        dataset_config: DatasetConfig,
        *,
        named_tasks: Mapping[str, PredictionTaskConfig] | None = None,
        default_task: str | None = None,
    ) -> DatasetConfig:
        """Resolve a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to resolve, e.g. `"argoverse1"` or `"vod"`.
        dataset_config : DatasetConfig
            The full configuration before resolution.
        named_tasks : Mapping[str, PredictionTaskConfig] or None
            Descriptor-owned tasks available to named selections.
        default_task : str or None
            Descriptor-owned task name to apply before authored overrides.
        """
        if default_task is not None:
            dataset_config = _resolve_named_task(
                dataset=dataset,
                name=default_task,
                config=dataset_config,
                named_tasks=named_tasks,
            )
        dataset_config = self._apply_entry(
            entry=self.defaults,
            target=dataset_config,
            context="defaults",
        )
        resolved = self._apply_entry(
            entry=self.datasets.get(dataset) if self.datasets else None,
            target=dataset_config,
            context=f"dataset '{dataset}'",
        )
        selection = self.task_selection_for(dataset)
        if not isinstance(selection, str):
            return resolved
        return _resolve_named_task(
            dataset=dataset,
            name=selection,
            config=resolved,
            named_tasks=named_tasks,
        )

    def _apply_entry(
        self,
        *,
        entry: DatasetConfigEntry | None,
        target: DatasetConfig,
        context: str,
    ) -> DatasetConfig:
        if entry is None:
            return target

        for use in entry.uses or ():
            profile = self.profiles.get(use)
            if profile is None:
                msg = f"Profile '{use}' not found for {context}"
                raise ConfigurationError(msg)
            target = profile.merge_into(target)

        return entry.to_dataset_config_patch().merge_into(target)


def _resolve_named_task(
    *,
    dataset: str,
    name: str,
    config: DatasetConfig,
    named_tasks: Mapping[str, PredictionTaskConfig] | None,
) -> DatasetConfig:
    if named_tasks is None:
        msg = (
            f"Named task '{name}' for dataset '{dataset}' requires the "
            "descriptor's `named_tasks` mapping."
        )
        raise ConfigurationError(msg)
    task = named_tasks.get(name)
    if task is None:
        available = ", ".join((*sorted(named_tasks), "none"))
        msg = f"Unknown task '{name}' for dataset {dataset}. Available tasks: {available}."
        raise ConfigurationError(msg)
    return DatasetConfig.model_validate(config.model_dump(mode="python") | {"task": task})
