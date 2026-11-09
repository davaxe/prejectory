"""Typer application for the optional Prejectory CLI."""

# ruff: file-ignore[import-outside-top-level]
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from pydantic import ValidationError
from rich import print as rprint
from rich.panel import Panel

from prejectory import __version__
from prejectory.core.categories import DatasetSplit
from prejectory.core.errors import (
    CliError,
    ConfigurationError,
    DatasetNotFoundError,
    MissingOptionalDependencyError,
    PrejectoryError,
    SplitError,
    cli_usage_error,
)
from prejectory.io.base import StorageBackend
from prejectory.runtime.cli.formatting import (
    PLAN_NOTICE,
    build_available_datasets_table,
    build_dataset_inspect_tables,
    build_processing_summary_table,
    build_split_support_tables,
)
from prejectory.runtime.cli.inputs import config_overrides
from prejectory.runtime.cli.register import register_custom_datasets

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from prejectory.runtime.types import ExecutionPlan


app: typer.Typer = typer.Typer(help="Trajectory data processing toolkit.", no_args_is_help=True)

# Typer currently requires direct typing objects rather than PEP 695 alias objects.
ReadStrategy = Literal["all", "native"]
AssignStrategy = Literal["none", "preserve-native", "scene", "source", "time", "shuffled-time"]

InputDir = Annotated[
    Path,
    typer.Option("--input", "-i", help="Directory containing the raw dataset."),
]
OutputDir = Annotated[
    Path,
    typer.Option("--output", "-o", help="Directory to save the processed dataset."),
]
Read = Annotated[
    ReadStrategy | None,
    typer.Option("--read", help="Read mode to use.", show_default=False, rich_help_panel="Read"),
]
ReadSplit = Annotated[
    list[DatasetSplit] | None,
    typer.Option(
        "--read-split",
        help="Dataset-defined partition(s) to read when --read native is selected.",
        show_default=False,
        rich_help_panel="Read",
    ),
]
Assign = Annotated[
    AssignStrategy | None,
    typer.Option(
        "--assign",
        help="Output assignment mode to use.",
        show_default=False,
        rich_help_panel="Assign splits",
    ),
]
Config = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to the optional configuration file."),
]
Jobs = Annotated[
    int | None,
    typer.Option(
        "--jobs",
        "-j",
        help="Worker count override. Values greater than 1 enable parallel execution.",
    ),
]
Progress = Annotated[
    bool,
    typer.Option("--progress/--no-progress", help="Show progress during processing."),
]
Limit = Annotated[
    int | None,
    typer.Option("--limit", "-l", help="Limit the number of scenes to process."),
]
Seed = Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Random seed used by split assignment and other randomized operations.",
    ),
]
StorageBackendOption = Annotated[
    str,
    typer.Option("--storage-backend", "--sb", help="Storage backend for processed data."),
]
TrajectorySchema = Annotated[
    str | None,
    typer.Option("--scene-schema", help="Scene schema to persist in exported output."),
]
SplitRatio = Annotated[
    tuple[float, float, float] | None,
    typer.Option(
        "--ratio",
        help="Train/val/test ratio used by scene, source, time, and shuffled-time assignment.",
        rich_help_panel="Assign splits",
    ),
]
SplitGap = Annotated[
    int | None,
    typer.Option(
        "--gap",
        help="Gap inserted between time partitions.",
        rich_help_panel="Assign splits",
    ),
]
SplitSegments = Annotated[
    int | None,
    typer.Option(
        "--segments",
        help="Number of contiguous temporal segments used by shuffled-time assignment.",
        rich_help_panel="Assign splits",
    ),
]
Force = Annotated[
    bool,
    typer.Option("--force", "-f", help="Force processing without confirmation."),
]
Overwrite = Annotated[
    bool,
    typer.Option(
        "--overwrite/--no-overwrite",
        help="Replace an existing non-empty output directory.",
    ),
]
DatasetName = Annotated[
    str,
    typer.Argument(help="The name of the dataset to apply the command to."),
]
Plan = Annotated[
    bool,
    typer.Option("--plan/--no-plan", help="Show the processing plan without executing it."),
]
IncludeMap = Annotated[
    bool | None,
    typer.Option("--include-map/--no-map", help="Include the map (if available)."),
]
DatasetModule = Annotated[
    list[str] | None,
    typer.Option(
        "--dataset-module",
        help=("Import a module that registers custom datasets. Can be provided more than once."),
        show_default=False,
    ),
]


def _version_callback(*, version: bool) -> None:
    if version:
        rprint(f"prejectory version: {__version__}")
        raise typer.Exit


@app.callback()
def global_options(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
    dataset_module: DatasetModule = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", envvar="PREJECTORY_LOG_LEVEL", help="Python logging level."),
    ] = "WARNING",
) -> None:
    """Global CLI options."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    _ = version
    register_custom_datasets(dataset_module)


@app.command()
def process(
    *,
    dataset: DatasetName,
    input_dir: InputDir,
    output_dir: OutputDir,
    read: Read = None,
    read_split: ReadSplit = None,
    assign: Assign = None,
    config: Config = None,
    jobs: Jobs = None,
    progress: Progress = True,
    limit: Limit = None,
    seed: Seed = None,
    storage_backend: StorageBackendOption = StorageBackend.PICKLE.value,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    force: Force = False,
    overwrite: Overwrite = False,
    plan_mode: Plan = False,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from prejectory.runtime.api import execute_plan

    plan = _run_cli_action(
        lambda: _resolve_cli_plan(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            storage_backend=storage_backend,
            config=config,
            read=read,
            read_split=read_split,
            assign=assign,
            jobs=jobs,
            trajectory_schema=trajectory_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            include_map=include_map,
            limit=limit,
            seed=seed,
            overwrite=overwrite,
            input_dir_exists=not plan_mode,
        ),
    )

    if plan_mode:
        rprint(PLAN_NOTICE)
        rprint(build_processing_summary_table(plan))
        return

    if not force:
        rprint("\n", build_processing_summary_table(plan))
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    _ = _run_cli_action(lambda: execute_plan(plan, show_progress=progress))


@app.command()
def available(
    *,
    details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed information about datasets."),
    ] = True,
) -> None:
    """[bold]List available datasets[/bold]."""
    from prejectory.datasets import list_datasets
    from prejectory.datasets.registry import get_dataset

    descriptors = _run_cli_action(lambda: [get_dataset(dataset) for dataset in list_datasets()])
    rprint("\n", build_available_datasets_table(descriptors, details=details))


@app.command()
def inspect(dataset: DatasetName) -> None:
    """[bold]Inspect details for a specified dataset[/bold]."""
    from prejectory.datasets.registry import get_dataset

    descriptor = _run_cli_action(lambda: get_dataset(dataset))
    _print_tables(build_dataset_inspect_tables(descriptor))


@app.command()
def show_config(
    dataset: DatasetName,
    read: Read = None,
    read_split: ReadSplit = None,
    assign: Assign = None,
    config: Config = None,
    jobs: Jobs = None,
    storage_backend: StorageBackendOption = StorageBackend.PICKLE.value,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Show the resolved configuration for a specified dataset[/bold]."""
    plan = _run_cli_action(
        lambda: _resolve_cli_plan(
            dataset=dataset,
            input_dir=Path(),
            output_dir=Path.cwd().parent / "__prejectory_config_output__",
            storage_backend=storage_backend,
            config=config,
            read=read,
            read_split=read_split,
            assign=assign,
            jobs=jobs,
            trajectory_schema=trajectory_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            include_map=include_map,
            input_dir_exists=False,
        ),
    )
    rprint("\n[bold]Resolved Config:[/bold]")
    rprint(plan.config)


@app.command()
def split_support(dataset: DatasetName) -> None:
    """[bold]Show the split support for a specified dataset[/bold]."""
    from prejectory.datasets.registry import get_dataset

    descriptor = _run_cli_action(lambda: get_dataset(dataset))
    _print_tables(build_split_support_tables(descriptor))


def main() -> None:
    """Run the optional Prejectory CLI application."""
    try:
        app()
    except CliError as exc:
        rprint(Panel(str(exc), title="Error", border_style="red"))
        raise SystemExit(exc.exit_code) from None


def _print_tables(tables: Iterable[object]) -> None:
    for index, table in enumerate(tables):
        rprint("\n", table) if index == 0 else rprint(table)


def _run_cli_action[T](action: Callable[[], T]) -> T:
    try:
        return action()
    except CliError:
        raise
    except FileNotFoundError as exc:
        raise _file_error(exc) from exc
    except ValidationError as exc:
        raise cli_usage_error(_format_validation_error(exc)) from exc
    except (ConfigurationError, DatasetNotFoundError, SplitError) as exc:
        raise cli_usage_error(str(exc)) from exc
    except MissingOptionalDependencyError as exc:
        raise CliError(str(exc)) from exc
    except PrejectoryError as exc:
        raise CliError(str(exc)) from exc


def _file_error(exc: FileNotFoundError) -> CliError:
    if exc.filename is None:
        return cli_usage_error(str(exc))
    return CliError(f"Could not open file: {exc.filename}")


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "value"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


def _resolve_cli_plan(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    storage_backend: StorageBackend | str,
    config: Path | None,
    read: ReadStrategy | None,
    read_split: list[DatasetSplit] | None,
    assign: AssignStrategy | None,
    jobs: int | None,
    trajectory_schema: str | None,
    ratio: tuple[float, float, float] | None,
    gap: int | None,
    segments: int | None,
    include_map: bool | None,
    limit: int | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    input_dir_exists: bool = True,
) -> ExecutionPlan:
    from prejectory.runtime import ExecutionRequest, resolve_request

    return resolve_request(
        ExecutionRequest(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            limit=limit,
            seed=seed,
            overwrite=overwrite,
            storage_backend=storage_backend,
            config=config,
            overrides=config_overrides(
                read_strategy=read,
                read_split=read_split,
                assign_strategy=assign,
                jobs=jobs,
                trajectory_schema=trajectory_schema,
                ratio=ratio,
                gap=gap,
                segments=segments,
            ),
            include_map=include_map,
            input_dir_exists=input_dir_exists,
        ),
    )
