# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import pytest

from prejectory.config import DatasetConfigPatch, RuntimePatch
from prejectory.core.errors import (
    CliError,
    ConfigurationError,
    DatasetNotFoundError,
    UnsupportedStorageBackendError,
)
from prejectory.datasets import (
    DatasetFeatureSupport,
    DatasetTemporalSupport,
    DatasetWindowingSupport,
    FrameBounds,
    list_datasets,
)
from prejectory.datasets.registry import _REGISTRY, dataset_names_by_id
from prejectory.io import PredictionBounds, StorageBackend, read_manifest
from prejectory.io.backends.null import NullWriter
from prejectory.io.base import WorkerWriterProvider
from prejectory.io.readers import PickleReader
from prejectory.processing.screening.agent import AgentRequireFrames
from prejectory.runtime import ExecutionRequest, OutputTransform, execute_request, resolve_request
from prejectory.runtime.executor import open_executor
from prejectory.runtime.processor import RuntimeProcessor
from tests.support import (
    DemoOptions,
    cleanup_demo_descriptor,
    demo_descriptor,
    stale_kinematics_demo_descriptor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from prejectory.core.scene import Scene
    from prejectory.datasets import DatasetDescriptor
    from prejectory.io.records import SceneRecord


def _request(tmp_path: Path, **kwargs: object) -> ExecutionRequest:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base: dict[str, object] = {
        "dataset": "demo",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "storage_backend": StorageBackend.NULL,
    }
    base.update(kwargs)
    return ExecutionRequest.model_validate(base)


def _cli_app_and_runner() -> tuple[Any, Any]:
    pytest.importorskip("typer")
    pytest.importorskip("rich")

    from typer.testing import CliRunner  # ruff: ignore[import-outside-top-level]

    import prejectory.runtime.cli.app as cli_app  # ruff: ignore[import-outside-top-level]

    return cli_app.app, CliRunner()


def _patch_descriptor(monkeypatch: pytest.MonkeyPatch, descriptor: DatasetDescriptor) -> None:
    def provider(_name: str) -> DatasetDescriptor:
        return descriptor

    monkeypatch.setattr("prejectory.runtime.api.get_dataset", provider)


def _patch_get_demo_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_descriptor(monkeypatch, demo_descriptor())


def _create_null_writer(_worker_id: int) -> NullWriter:
    return NullWriter()


class FailingWriter:
    def write(self, scene: Scene) -> None:  # ruff: ignore[no-self-use]
        _ = scene
        msg = "intentional writer failure"
        raise RuntimeError(msg)

    def finish_local(self) -> None:  # ruff: ignore[no-self-use]
        return


def _create_failing_writer(_worker_id: int) -> FailingWriter:
    return FailingWriter()


def test_resolve_request_builds_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    plan = resolve_request(_request(tmp_path, include_map=False))

    assert plan.dataset == "demo"
    assert plan.storage_backend == StorageBackend.NULL
    dataset_options = cast("DemoOptions", plan._loader.loader_options)
    assert dataset_options.batch_size == 2
    assert not plan.include_map
    assert plan.effective_horizon_frames == 3
    assert plan.effective_prediction_bounds == PredictionBounds(2, 3)


def test_resolve_request_rejects_unknown_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    with pytest.raises(UnsupportedStorageBackendError, match="Unsupported storage backend"):
        _ = resolve_request(_request(tmp_path, storage_backend="bad-backend"))


def test_resolve_request_rejects_lane_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    config_path = tmp_path / "prejectory.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support lane-change sampling"):
        _ = resolve_request(_request(tmp_path, config=config_path))


def test_resolve_request_requires_window_for_lane_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(
        demo_descriptor(),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "prejectory.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes]
window = { op = "clear" }

[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires window sampling"):
        _ = resolve_request(_request(tmp_path, config=config_path))


def test_resolve_request_rejects_long_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=2, confidence="documented"),
            windowing=DatasetWindowingSupport(enabled_by_default=True),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)

    with pytest.raises(ConfigurationError, match="supports windows up to 2 source frames"):
        _ = resolve_request(_request(tmp_path))


def test_resampling_recomputes_native_kinematics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = stale_kinematics_demo_descriptor()
    _patch_descriptor(monkeypatch, descriptor)
    plan = resolve_request(_request(tmp_path, dataset=descriptor.name))
    assert plan.effective_prediction_bounds == PredictionBounds(1, 2)
    loader = descriptor.build_loader(root=plan.input_dir, request=plan._loader)
    processor = RuntimeProcessor.from_plan(plan, loader)
    source = next(iter(processor.iter_sources()))
    candidate = next(iter(processor.iter_candidates(source)))

    scene = processor.materialize(candidate, scene_number=0)

    assert scene.horizon_frames == 2
    assert scene.frame["frame"].to_list() == [0, 1]
    assert scene.frame["vx"].to_list() == pytest.approx([1.0, 1.0])
    assert scene.frame["vy"].to_list() == pytest.approx([0.0, 0.0])
    assert scene.frame["ax"].to_list() == pytest.approx([0.0, 0.0])
    assert set(plan.manifest().derived_features) == {"vx", "vy", "ax", "ay", "yaw"}


def test_resolve_request_rejects_window_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=10, confidence="documented"),
            windowing=DatasetWindowingSupport(
                enabled_by_default=True,
                supported_policies=("strict",),
            ),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.window]
step = 1
policy = "partial"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support window policy 'partial'"):
        _ = resolve_request(_request(tmp_path, config=config_path))


def test_execute_request_surfaces_unknown_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path, dataset="this-dataset-does-not-exist")

    with pytest.raises(DatasetNotFoundError):
        _ = execute_request(request)


def test_execute_request_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    request = _request(tmp_path)
    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.storage_backend == StorageBackend.NULL
    assert result.stats.processed_sources == 1

    manifest = read_manifest(result.output_dir)
    assert manifest.storage_backend == "null"
    assert manifest.dataset_names == ("demo",)
    assert manifest.source_trajectory_schema_fields == (
        "frame",
        "id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "yaw",
        "agent_category",
    )
    assert manifest.horizon_frames == 3
    assert manifest.prediction_task is not None
    assert manifest.prediction_task.prediction_origin == 2
    assert manifest.prediction_task.prediction_end == 3


def test_execute_request_rejects_non_empty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    request = _request(tmp_path)
    request.output_dir.mkdir()
    marker = request.output_dir / "old-data"
    _ = marker.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _ = execute_request(request, show_progress=False)

    assert marker.exists()


def test_execute_request_overwrites_output_when_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    request = _request(tmp_path, overwrite=True)
    request.output_dir.mkdir()
    marker = request.output_dir / "old-data"
    _ = marker.write_text("stale", encoding="utf-8")

    result = execute_request(request, show_progress=False)

    assert not marker.exists()
    assert read_manifest(result.output_dir).dataset == "demo"


def test_builtin_manifest_uses_global_dataset_name_table(tmp_path: Path) -> None:
    request = ExecutionRequest(
        dataset="a43",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        storage_backend=StorageBackend.NULL,
        input_dir_exists=False,
    )
    plan = resolve_request(request)

    assert plan.manifest().dataset_names == dataset_names_by_id()


def test_builtin_benchmark_task_is_selected_by_default(tmp_path: Path) -> None:
    request = ExecutionRequest(
        dataset="argoverse1",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        storage_backend=StorageBackend.NULL,
        input_dir_exists=False,
    )
    plan = resolve_request(request)
    assert plan.selected_task == "benchmark"
    assert plan.effective_prediction_bounds == PredictionBounds(20, 50)
    manifest_task = plan.manifest().prediction_task
    assert manifest_task is not None
    assert manifest_task.name == "benchmark"
    assert manifest_task.prediction_origin == 20
    assert manifest_task.prediction_end == 50


def test_resolve_request_rejects_unknown_task(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.argoverse1]
task = "missing"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Unknown task 'missing'"):
        _ = resolve_request(
            ExecutionRequest(
                dataset="argoverse1",
                input_dir=tmp_path / "input",
                output_dir=tmp_path / "output",
                config=config_path,
                input_dir_exists=False,
            ),
        )


def test_project_can_select_named_task_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.argoverse1]
task = "benchmark"
""",
        encoding="utf-8",
    )
    plan = resolve_request(
        ExecutionRequest(
            dataset="argoverse1",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            config=config_path,
            input_dir_exists=False,
        ),
    )

    assert plan.selected_task == "benchmark"
    assert plan.effective_prediction_bounds == PredictionBounds(20, 50)


def test_project_can_disable_default_task(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.argoverse1]
task = "none"
""",
        encoding="utf-8",
    )
    plan = resolve_request(
        ExecutionRequest(
            dataset="argoverse1",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            config=config_path,
            input_dir_exists=False,
        ),
    )

    assert plan.selected_task is None
    assert plan.effective_prediction_bounds is None
    assert plan.manifest().prediction_task is None


def test_inline_dataset_task_replaces_default_named_task(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.argoverse1.task]
prediction_origin = 10
prediction_end = 30
""",
        encoding="utf-8",
    )
    plan = resolve_request(
        ExecutionRequest(
            dataset="argoverse1",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            config=config_path,
            input_dir_exists=False,
        ),
    )
    assert plan.selected_task is None
    assert plan.effective_prediction_bounds == PredictionBounds(10, 30)
    manifest_task = plan.manifest().prediction_task
    assert manifest_task is not None
    assert manifest_task.name is None
    endpoint = plan._loader.screening
    assert endpoint is not None
    rule = endpoint.agents["prediction_history_endpoint"]
    assert isinstance(rule, AgentRequireFrames)
    assert rule.frames == frozenset({9})


def test_execute_request_applies_record_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset_id": record.dataset_id,
            "feature_shape": record.features.shape,
        }

    output_transform = OutputTransform(format_id="test.custom", record_transform=transform)
    request = _request(
        tmp_path,
        storage_backend=StorageBackend.PICKLE,
        output_transform=output_transform,
    )

    result = execute_request(request)
    record = cast("dict[str, object]", PickleReader(result.output_dir, record_type=dict)[0])

    assert record == {"scene_number": 0, "dataset_id": None, "feature_shape": (1, 3, 7)}


def test_execute_request_writes_custom_mds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "streaming",
        reason="Requires streaming package for custom MDS output record format",
    )
    from prejectory.io.readers import MDSReader  # ruff: ignore[import-outside-top-level]

    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset_id": -1 if record.dataset_id is None else record.dataset_id,
            "feature_shape": record.features.shape,
        }

    output_transform = OutputTransform(
        format_id="test.custom",
        record_transform=transform,
        mds_columns={"scene_number": "int", "dataset_id": "int", "feature_shape": "json"},
    )
    request = _request(
        tmp_path,
        storage_backend=StorageBackend.MDS,
        output_transform=output_transform,
    )
    result = execute_request(request)
    reader = MDSReader(path=result.output_dir, convert_raw=dict)
    record = reader[0]
    assert record == {"scene_number": 0, "dataset_id": -1, "feature_shape": [1, 3, 7]}


def test_parallel_execution_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    _patch_get_demo_descriptor(monkeypatch)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=DatasetConfigPatch(runtime=RuntimePatch(jobs=2)),
    )

    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.stats.processed_sources == 1
    assert result.stats.candidate_scenes == 1
    assert result.stats.written_scenes == 1
    assert result.stats.split_counts["unsplit"] == 1

    manifest = read_manifest(output_dir)
    assert manifest.horizon_frames == 3
    assert manifest.prediction_task is not None
    assert manifest.prediction_task.prediction_origin == 2
    assert manifest.prediction_task.prediction_end == 3
    assert manifest.dataset_names == ("demo",)


def test_execute_request_reports_cleanup_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    request = _request(tmp_path, dataset="cleanup-demo")
    result = execute_request(request)

    assert result.cleanup_summary is not None
    assert result.cleanup_summary.overall.scene_count == 1
    assert result.cleanup_summary.overall.total_rows_removed == 3
    assert result.cleanup_summary.overall.total_agents_removed == 1
    assert result.cleanup_summary.overall.average_rows_removed_per_scene == pytest.approx(3.0)
    assert result.cleanup_summary.overall.min_rows_removed_per_scene == 3
    assert result.cleanup_summary.overall.max_rows_removed_per_scene == 3
    assert "trim_unimportant" in result.cleanup_summary.by_rule
    rule_summary = result.cleanup_summary.by_rule["trim_unimportant"]
    assert rule_summary.total_rows_removed == 3
    assert rule_summary.total_agents_removed == 1


def test_parallel_execution_reports_cleanup_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="cleanup-demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=DatasetConfigPatch(runtime=RuntimePatch(jobs=2)),
    )

    result = execute_request(request)

    assert result.cleanup_summary is not None
    assert result.cleanup_summary.overall.total_rows_removed == 3
    assert result.cleanup_summary.overall.total_agents_removed == 1
    assert result.cleanup_summary.by_rule["trim_unimportant"].scene_count == 1


@pytest.mark.parametrize("jobs", [None, 2], ids=["sequential", "parallel"])
def test_execution_progress_reports_cleanup_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs: int | None,
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    request_kwargs: dict[str, object] = {"dataset": "cleanup-demo"}
    if jobs is not None:
        request_kwargs["overrides"] = DatasetConfigPatch(runtime=RuntimePatch(jobs=jobs))

    plan = resolve_request(_request(tmp_path, **request_kwargs))
    writer_provider = WorkerWriterProvider(_create_null_writer)

    with open_executor(plan) as executor:
        progress = executor.execute(writer_provider)

    assert progress.cleanup.rows_total == 6
    assert progress.cleanup.rows_removed == 3
    assert progress.cleanup.agents_total == 2
    assert progress.cleanup.agents_removed == 1


def test_failed_writer_is_not_counted_as_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    plan = resolve_request(_request(tmp_path))
    writer_provider = WorkerWriterProvider(_create_failing_writer)  # pyright: ignore[reportArgumentType]

    with open_executor(plan) as executor:
        with pytest.raises(RuntimeError, match="intentional writer failure"):
            _ = executor.execute(writer_provider)
        progress = executor.snapshot()

    assert progress.stats.written_scenes == 0
    assert progress.stats.split_counts["unsplit"] == 0


@pytest.mark.parametrize(
    "name",
    ["process", "available", "inspect", "show-config", "split-support"],
    ids=["process", "available", "inspect", "show-config", "split-support"],
)
def test_cli_commands_smoke(tmp_path: Path, name: str) -> None:
    app, runner = _cli_app_and_runner()
    for dataset_name in list_datasets():
        output_dir = tmp_path / "cli-output"
        args_by_command: dict[str, list[str]] = {
            "process": [
                "process",
                dataset_name,
                "--input",
                ".",
                "--output",
                str(output_dir),
                "--plan",
            ],
            "available": ["available", "--no-details"],
            "inspect": ["inspect", dataset_name],
            "show-config": ["show-config", dataset_name],
            "split-support": ["split-support", dataset_name],
        }
        args = args_by_command[name]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{name} failed: {result.output}"


def test_cli_help_smoke() -> None:
    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0


def test_process_cli_has_no_task_overrides() -> None:
    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["process", "--help"])

    assert result.exit_code == 0
    assert "--task" not in result.output
    assert "--prediction-origin" not in result.output
    assert "--prediction-end" not in result.output


def test_inspect_reports_temporal_support() -> None:
    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["inspect", "argoverse1"])

    assert result.exit_code == 0, result.output
    assert "Configured horizon" in result.output
    assert "Sliding windows" in result.output
    assert "Source bounds" in result.output
    assert "Configured horizon fits" in result.output
    assert "Supported policies" in result.output


def test_cli_imports_dataset_module_before_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "custom_datasets.py"
    _ = module_path.write_text(
        (
            "from dataclasses import replace\n"
            "from tests.support import demo_descriptor\n"
            "\n"
            "\n"
            "def register_prejectory_datasets():\n"
            '    return replace(demo_descriptor(), name="cli_demo")\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _ = _REGISTRY.pop("cli_demo", None)

    try:
        app, runner = _cli_app_and_runner()
        available_result = runner.invoke(
            app,
            ["--dataset-module", "custom_datasets", "available", "--no-details"],
        )
        inspect_result = runner.invoke(
            app,
            ["--dataset-module", "custom_datasets", "inspect", "cli_demo"],
        )
    finally:
        _ = _REGISTRY.pop("cli_demo", None)

    assert available_result.exit_code == 0, available_result.output
    assert "cli_demo" in available_result.output
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "cli_demo" in inspect_result.output


@pytest.mark.parametrize(
    ("module_body", "expected"),
    [
        ("register_prejectory_datasets = 1\n", "non-callable"),
        ("def register_prejectory_datasets():\n    return 1\n", "unsupported value"),
        ("def register_prejectory_datasets():\n    return [1]\n", "returned int"),
    ],
)
def test_cli_rejects_invalid_dataset_module_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_body: str,
    expected: str,
) -> None:
    module_path = tmp_path / "bad_datasets.py"
    _ = module_path.write_text(module_body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    _ = sys.modules.pop("bad_datasets", None)

    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["--dataset-module", "bad_datasets", "available", "--no-details"])

    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert isinstance(result.exception, CliError)
    assert expected in message
