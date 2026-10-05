"""Behavioral coverage for the public Python workflow."""

from __future__ import annotations

import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] - Isolated import smoke test.
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from prejectory import ExecutionRequest, open_dataset, plan, run
from prejectory.config import (
    Clear,
    DatasetConfigEntry,
    DatasetConfigPatch,
    DictPatch,
    MapPatch,
    OutputPatch,
    PredictionTaskConfig,
    ProjectConfig,
    RuntimePatch,
    SceneAssign,
    ScenesPatch,
    SplitWeights,
)
from prejectory.core.errors import ConfigurationError, MissingPredictionBoundsError
from prejectory.io import PredictionBounds, SceneRecord, read_manifest, write_manifest
from prejectory.io.readers import PickleReader
from prejectory.runtime import OutputTransform, Progress
from tests.support import demo_descriptor

if TYPE_CHECKING:
    from collections.abc import Callable

    from prejectory.datasets import DatasetDescriptor


def _lookup(descriptor: DatasetDescriptor) -> Callable[[str], DatasetDescriptor]:
    def lookup(_name: str) -> DatasetDescriptor:
        return descriptor

    return lookup


@pytest.fixture
def run_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExecutionRequest:
    descriptor = demo_descriptor()
    monkeypatch.setattr("prejectory.runtime.api.get_dataset", _lookup(descriptor))
    source = tmp_path / "input"
    source.mkdir()
    return ExecutionRequest(dataset="demo", input_dir=source, output_dir=tmp_path / "output")


def test_configuration_has_one_merge_path(run_request: ExecutionRequest) -> None:
    project = ProjectConfig(
        defaults=DatasetConfigEntry(output=OutputPatch(precision="float64")),
        datasets={"demo": DatasetConfigEntry(map=MapPatch(min_distance=4))},
    )
    overrides = DatasetConfigPatch(
        scenes=ScenesPatch(horizon_frames=2),
        map=MapPatch(min_distance=None),
        loader_options=DictPatch.model_validate({"batch_size": 7}),
        screening=Clear(),
    )
    resolved = plan(
        run_request.model_copy(update={"config": project, "overrides": overrides, "task": None})
    )
    assert resolved.config.output.precision == "float64"
    assert resolved.config.scenes.horizon_frames == 2
    assert resolved.config.map.min_distance is None
    assert resolved.config.loader_options == {"batch_size": 7, "use_cache": False}
    assert resolved.config.task is None
    assert resolved.config.screening is None
    assert resolved.effective_prediction_bounds is None
    assert not run_request.output_dir.exists()


def test_project_resolves_name_or_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = demo_descriptor()
    monkeypatch.setattr("prejectory.datasets.registry.get_dataset", _lookup(descriptor))
    project = ProjectConfig(defaults=DatasetConfigEntry(output=OutputPatch(precision="float64")))
    by_name = project.resolve_dataset_config("demo")
    assert by_name == project.resolve_dataset_config(descriptor)
    assert by_name.output.precision == "float64"
    assert descriptor.default_config.output.precision == "float32"


def test_named_task_inheritance_and_explicit_disable(
    run_request: ExecutionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = PredictionTaskConfig(prediction_origin=1, prediction_end=3)
    descriptor = replace(demo_descriptor(), tasks={"forecast": task}, default_task="forecast")
    monkeypatch.setattr("prejectory.runtime.api.get_dataset", _lookup(descriptor))
    assert plan(run_request).selected_task == "forecast"
    assert plan(run_request.model_copy(update={"task": None})).config.task is None
    chosen = plan(run_request.model_copy(update={"task": "forecast"}))
    assert chosen.effective_prediction_bounds == PredictionBounds(1, 3)
    with pytest.raises(ConfigurationError, match="Unknown task"):
        _ = plan(run_request.model_copy(update={"task": "missing"}))


def test_run_executes_inspected_plan_after_cwd_and_config_change(
    run_request: ExecutionRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = run_request  # Installs the synthetic dataset and creates its input directory.
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    _ = config.write_text('[defaults.output]\nprecision = "float64"\n', encoding="utf-8")
    relative = ExecutionRequest(
        dataset="demo", input_dir=Path("input"), output_dir=Path("output"), config="config.toml"
    )
    resolved = plan(relative)
    assert resolved.input_dir.is_absolute()
    assert resolved.output_dir.is_absolute()
    assert resolved.config.output.precision == "float64"
    _ = config.write_text("invalid TOML", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = run(resolved)
    assert result.written_scenes == result.stats.written_scenes == 1
    dataset = open_dataset(str(result.output_dir))
    assert dataset.manifest.precision == "float64"
    assert dataset.splits == ("unsplit",)
    assert dataset.manifest.split_counts == {"unsplit": 1}
    assert isinstance(dataset[0], SceneRecord)
    assert dataset[-1].scene_number == 0
    assert len(list(dataset)) == 1
    assert "demo:" in resolved.summary()
    with pytest.raises(IndexError):
        _ = dataset[1]


@pytest.mark.parametrize("backend", ["pickle", "mds", "zarr"])
def test_open_dataset_detects_backend_and_selects_splits(
    run_request: ExecutionRequest, backend: str
) -> None:
    if backend == "mds":
        pytest.importorskip("streaming")
    elif backend == "zarr":
        pytest.importorskip("zarr")
    result = run(
        run_request.model_copy(
            update={
                "storage_backend": backend,
                "overrides": DatasetConfigPatch(
                    assign=SceneAssign(ratio=SplitWeights(train=1, val=1))
                ),
                "seed": 0,
            }
        )
    )
    dataset = open_dataset(result.output_dir)
    assert set(dataset.splits) == {"train", "val"}
    assert len(dataset) == result.written_scenes
    assert len(list(dataset)) == result.written_scenes
    assert sum(len(open_dataset(result.output_dir, split=s)) for s in dataset.splits) == len(
        dataset
    )
    with pytest.raises(ConfigurationError, match="Available splits"):
        _ = open_dataset(result.output_dir, split="unsplit")


def test_reader_errors_distinguish_missing_and_empty(run_request: ExecutionRequest) -> None:
    with pytest.raises(FileNotFoundError):
        _ = PickleReader(run_request.output_dir)
    result = run(run_request)
    manifest = read_manifest(result.output_dir)
    write_manifest(result.output_dir, replace(manifest, split_counts={"unsplit": 2}))
    with pytest.raises(ConfigurationError, match="manifest declares 2"):
        _ = open_dataset(result.output_dir)


def custom_record(record: SceneRecord) -> dict[str, int]:
    return {"number": record.scene_number}


def fail_record(record: SceneRecord) -> dict[str, int]:
    _ = record
    msg = "transform failed"
    raise RuntimeError(msg)


@pytest.mark.parametrize("backend", ["pickle", "mds"])
def test_custom_payload_requires_explicit_decoder(
    run_request: ExecutionRequest, backend: str
) -> None:
    if backend == "mds":
        pytest.importorskip("streaming")
    transform = OutputTransform(
        record_transform=custom_record,
        format_id="example.number",
        format_version=2,
        mds_columns={"number": "int"} if backend == "mds" else None,
    )
    result = run(
        run_request.model_copy(update={"storage_backend": backend, "output_transform": transform})
    )
    manifest = read_manifest(result.output_dir)
    assert (manifest.payload_format, manifest.payload_version) == ("example.number", 2)
    with pytest.raises(ConfigurationError, match="explicit decoder"):
        _ = open_dataset(result.output_dir)
    numbers = open_dataset(result.output_dir, decoder=lambda raw: int(raw["number"]))
    assert list(numbers) == [0]
    assert numbers[0] == 0


@pytest.mark.parametrize("jobs", [1, 2])
def test_failed_replacement_preserves_previous_output(
    run_request: ExecutionRequest, jobs: int
) -> None:
    first = run(run_request)
    before = read_manifest(first.output_dir)
    failed = run_request.model_copy(
        update={
            "overwrite": True,
            "overrides": DatasetConfigPatch(runtime=RuntimePatch(jobs=jobs)),
            "output_transform": OutputTransform(
                record_transform=fail_record, format_id="example.failure"
            ),
        }
    )
    with pytest.raises(RuntimeError, match="transform failed"):
        _ = run(failed)
    assert read_manifest(first.output_dir) == before
    assert len(open_dataset(first.output_dir)) == 1
    assert not list(first.output_dir.parent.glob(f".{first.output_dir.name}-*"))


def test_invalid_mds_transform_fails_during_plan(run_request: ExecutionRequest) -> None:
    invalid = run_request.model_copy(
        update={
            "storage_backend": "mds",
            "output_transform": OutputTransform(
                record_transform=custom_record, format_id="example.number"
            ),
        }
    )
    with pytest.raises(ConfigurationError, match="mds_columns"):
        _ = plan(invalid)
    assert not run_request.output_dir.exists()


def test_zarr_rejects_custom_payload_during_plan(run_request: ExecutionRequest) -> None:
    pytest.importorskip("zarr")
    invalid = run_request.model_copy(
        update={
            "storage_backend": "zarr",
            "output_transform": OutputTransform(
                record_transform=custom_record,
                format_id="example.number",
            ),
        }
    )
    with pytest.raises(ConfigurationError, match="canonical SceneRecord"):
        _ = plan(invalid)
    assert not run_request.output_dir.exists()


def test_parallel_zarr_export_is_readable(run_request: ExecutionRequest) -> None:
    pytest.importorskip("zarr")
    result = run(
        run_request.model_copy(
            update={
                "storage_backend": "zarr",
                "overrides": DatasetConfigPatch(runtime=RuntimePatch(jobs=2)),
            }
        )
    )

    dataset = open_dataset(result.output_dir)
    assert len(dataset) == result.written_scenes == 1
    assert dataset[0].scene_number == 0


def test_progress_callback_is_observable_and_failure_does_not_publish(
    run_request: ExecutionRequest,
) -> None:
    callbacks: list[tuple[int, Progress]] = []

    def observe(snapshot: Progress) -> None:
        callbacks.append((threading.get_ident(), snapshot))

    result = run(run_request, on_progress=observe)
    assert callbacks[-1][1].stats.written_scenes == result.written_scenes
    assert len({ident for ident, _ in callbacks}) == 1
    assert callbacks[0][0] != threading.get_ident()

    def fail(snapshot: Progress) -> None:
        _ = snapshot
        msg = "observer failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="observer failed"):
        _ = run(run_request.model_copy(update={"overwrite": True}), on_progress=fail)
    assert len(open_dataset(result.output_dir)) == 1


def test_forecast_views_share_storage_and_use_explicit_bounds(
    run_request: ExecutionRequest,
) -> None:
    result = run(run_request.model_copy(update={"task": None}))
    record = open_dataset(result.output_dir)[0]
    with pytest.raises(MissingPredictionBoundsError):
        _ = record.forecast()
    forecast = record.forecast(PredictionBounds(1, 3))
    assert np.shares_memory(forecast.history_features, record.features)
    assert np.shares_memory(forecast.future_mask, record.valid_mask)
    assert forecast.observation_length == 1
    assert forecast.future_length == 2


def test_readme_python_quickstart_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    snippet = re.search(r"``python\n(.*?)\n``", readme, flags=re.DOTALL)
    assert snippet is not None
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/a43/raw").mkdir(parents=True)
    monkeypatch.setattr("prejectory.runtime.api.get_dataset", _lookup(demo_descriptor()))
    namespace: dict[str, Any] = {}
    exec(compile(snippet.group(1), "README.md", "exec"), namespace)  # ruff: ignore[exec-builtin]
    assert len(namespace["dataset"]) == namespace["result"].written_scenes == 1


def test_root_import_keeps_optional_frameworks_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import prejectory, sys; "
                "assert not {'torch', 'torch_geometric', 'streaming', 'rich'} & sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_plan_snapshots_mutable_configuration(
    run_request: ExecutionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = demo_descriptor()
    monkeypatch.setattr("prejectory.runtime.api.get_dataset", _lookup(descriptor))
    resolved = plan(run_request)
    assert descriptor.default_config.loader_options is not None
    descriptor.default_config.loader_options["batch_size"] = 99
    assert resolved.config.loader_options is not None
    assert resolved.config.loader_options["batch_size"] == 2


def test_manifest_without_partitions_is_not_silently_empty(run_request: ExecutionRequest) -> None:
    result = run(run_request)
    manifest = read_manifest(result.output_dir)
    write_manifest(result.output_dir, replace(manifest, split_counts={}))
    with pytest.raises(ConfigurationError, match="incomplete"):
        _ = open_dataset(result.output_dir)


def test_failed_publication_restores_previous_output(
    run_request: ExecutionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run(run_request)
    marker = result.output_dir / "previous-run"
    _ = marker.write_text("keep", encoding="utf-8")
    rename = Path.rename

    def fail_publication(path: Path, target: str | Path) -> Path:
        if path.name == "output" and path.parent.name.startswith(".output-"):
            msg = "publication failed"
            raise OSError(msg)
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="publication failed"):
        _ = run(run_request.model_copy(update={"overwrite": True}))
    assert marker.read_text(encoding="utf-8") == "keep"
    assert len(open_dataset(result.output_dir)) == 1
