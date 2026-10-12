from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

import prejectory
from prejectory import config, core, datasets, io, processing, runtime
from prejectory.config import DatasetConfigPatch, ProjectConfig, parse_config
from prejectory.core import AgentCategory, DatasetSplit, EdgeType, errors
from prejectory.core.maps import MapGraph, SharedMapGraph
from prejectory.io import (
    DatasetManifest,
    ForecastRecord,
    PredictionBounds,
    PredictionTaskManifest,
    StorageBackend,
    manifest_path,
    read_manifest,
    write_manifest,
)
from prejectory.io import adapters as io_adapters
from prejectory.io import readers as io_readers
from prejectory.io.base import DatasetWriter
from prejectory.runtime import (
    ExecutionPlan,
    ExecutionRequest,
    ExecutionResult,
    execute_plan,
    execute_request,
    resolve_request,
)

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCED_BLOCK = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)\n```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
SKIP_VALIDATE_MARKER = "<!-- no-validate -->"
PYTHON_INFO_STRINGS = {"py", "python"}


def test_root_namespace_is_small() -> None:
    assert set(prejectory.__all__) == {"ExecutionRequest", "plan", "run", "open_dataset"}


def test_main_namespaces_exposed() -> None:
    assert datasets is not None
    assert runtime is not None
    assert processing is not None
    assert io is not None
    assert datasets.DatasetTemporalSupport is not None
    assert datasets.DatasetWindowingSupport is not None
    assert datasets.FrameBounds is not None


def test_core_and_runtime_exports_present() -> None:
    assert AgentCategory is not None
    assert DatasetSplit is not None
    assert EdgeType is not None
    assert MapGraph is not None
    assert SharedMapGraph is not None
    assert ExecutionPlan is not None
    assert ExecutionRequest is not None
    assert ExecutionResult is not None
    assert execute_plan is not None
    assert resolve_request is not None
    assert execute_request is not None


def test_io_and_config_exports_present() -> None:
    assert DatasetManifest is not None
    assert PredictionBounds is not None
    assert PredictionTaskManifest is not None
    assert ForecastRecord is not None
    assert DatasetWriter is not None
    assert manifest_path is not None
    assert read_manifest is not None
    assert write_manifest is not None
    assert ProjectConfig is not None
    assert DatasetConfigPatch is not None
    assert parse_config is not None


def test_reader_and_adapter_exports_declared() -> None:
    assert DatasetWriter is not None
    assert io_readers.DatasetReader is not None
    assert "DatasetReader" in io_readers.__all__
    assert "MDSReaderInitArgs" in io_readers.__all__
    assert "IterableTorchSceneDataset" in io_adapters.__all__
    assert "IterableTorchForecastDataset" in io_adapters.__all__
    assert "TorchForecastDataset" in io_adapters.__all__
    assert "TorchForecastRecord" in io_adapters.__all__
    assert "IterableHeteroSceneDataset" in io_adapters.__all__
    assert "IterableHeteroForecastDataset" in io_adapters.__all__
    assert "HeteroForecastDataset" in io_adapters.__all__
    assert "HeteroSceneDataset" in io_adapters.__all__


def test_runtime_executors_not_root_exports() -> None:
    runtime_module = __import__("prejectory.runtime", fromlist=["ParallelExecutor"])
    assert not hasattr(runtime_module, "ParallelExecutor")
    assert not hasattr(runtime_module, "SequentialExecutor")


def test_documented_runtime_imports_match_api() -> None:
    assert ExecutionRequest is runtime.ExecutionRequest
    assert resolve_request is runtime.resolve_request
    assert execute_request is runtime.execute_request
    assert execute_plan is runtime.execute_plan
    assert not hasattr(runtime, "ProcessRequest")
    assert not hasattr(runtime, "process_dataset")
    assert not hasattr(runtime, "resolve_job")
    assert ExecutionRequest.model_fields["storage_backend"].default == StorageBackend.PICKLE


def test_canonical_exports_cover_public_workflows() -> None:
    assert prejectory.plan is runtime.plan is runtime.resolve_request
    assert prejectory.run is runtime.run
    assert prejectory.open_dataset is io.open_dataset
    assert datasets.DatasetSplitSupport is not None
    assert io.IterableDatasetReader is io_readers.IterableDatasetReader
    assert "to_torch_scene_record" in io_adapters.__all__
    assert runtime.ExecutionStats is not None
    assert runtime.Progress is not None
    assert issubclass(errors.ConfigurationError, errors.PrejectoryError)
    assert not hasattr(config, "RuntimeOverride")
    assert not hasattr(datasets, "MapConfig")
    assert not hasattr(core, "TRAJECTORY_SCHEMAS")
    assert not hasattr(io, "SplitSceneRecord")


def _documented_code_params(language: str | set[str]) -> list[tuple[str, str, str]]:
    languages = {language} if isinstance(language, str) else language
    params: list[tuple[str, str, str]] = []
    for doc_path in sorted(DOCS_ROOT.rglob("*.md")):
        relative_path = doc_path.relative_to(DOCS_ROOT).as_posix()
        text = doc_path.read_text(encoding="utf-8")
        for block_index, match in enumerate(FENCED_BLOCK.finditer(text), start=1):
            info = match.group("info").strip().split()
            prefix = text[: match.start()].rstrip()
            marker_line = prefix.rsplit("\n", maxsplit=1)[-1] if prefix else ""
            if not languages.intersection(info) or marker_line == SKIP_VALIDATE_MARKER:
                continue
            block_id = f"{relative_path}:block-{block_index}"
            params.append((block_id, relative_path, match.group("body").strip()))
    return params


def _documented_config_params() -> list[tuple[str, str, str]]:
    params = _documented_code_params("toml")
    assert params, "No TOML documentation examples found"
    return params


def _documented_python_params() -> list[tuple[str, str, str]]:
    params = _documented_code_params(PYTHON_INFO_STRINGS)
    assert params, "No Python documentation examples found"
    return params


@pytest.mark.parametrize(
    ("block_id", "relative_path", "snippet"),
    _documented_config_params(),
    ids=lambda value: value,
)
def test_documented_config_examples_validate(
    tmp_path: Path,
    block_id: str,
    relative_path: str,
    snippet: str,
) -> None:
    """Test that TOML code blocks in the docs can be parsed as valid config."""
    _ = block_id
    _ = relative_path
    config_path = tmp_path / "example.toml"
    _ = config_path.write_text(snippet + "\n", encoding="utf-8")
    _ = parse_config(config_path)


@pytest.mark.parametrize(
    ("block_id", "relative_path", "snippet"),
    _documented_python_params(),
    ids=lambda value: value,
)
def test_documented_python_examples_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    block_id: str,
    relative_path: str,
    snippet: str,
) -> None:
    """Test that Python code blocks in the docs run without error."""
    _ = relative_path
    example_path = tmp_path / f"{block_id.replace('/', '__').replace(':', '_')}.py"
    _ = example_path.write_text(snippet + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _ = runpy.run_path(str(example_path), run_name="__main__")


def test_config_file_parses() -> None:
    path = Path("prejectory.toml")
    _ = parse_config(path)


def test_markdown_links_resolve() -> None:
    failures: list[str] = []
    for doc_path in sorted(DOCS_ROOT.rglob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group("target").strip()
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", maxsplit=1)[0].strip()
            if not target:
                continue
            if not target.endswith(".md"):
                continue
            linked_path = (doc_path.parent / target).resolve()
            if not linked_path.is_file():
                source = doc_path.relative_to(DOCS_ROOT).as_posix()
                failures.append(f"{source}: missing link target {target}")

    assert not failures, "\n".join(failures)
