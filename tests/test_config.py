# ruff: file-ignore[private-member-access] - Internal plan/config consumers.
# pyright: reportPrivateUsage=false
# pyright: standard
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from prejectory.config import ProjectConfig, parse_config
from prejectory.config.models import (
    DatasetConfig,
    MapEdgeTypeRules,
    SceneExtentExtraction,
    ScreeningConfig,
    TrajectoryBufferExtraction,
)
from prejectory.core import AgentCategory
from prejectory.core.categories import DatasetSplit, EdgeType
from prejectory.core.errors import ConfigurationError
from prejectory.processing.columns import TrajectoryColumns
from prejectory.processing.screening import ScreeningRuleSet, agent, cleanup, scene
from prejectory.processing.screening.screen import screen_data
from prejectory.runtime.cli.inputs import config_overrides
from tests.support import inherited_optional_blocks_descriptor

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, body: str) -> Path:
    config_path = path / "config.toml"
    _ = config_path.write_text(body.strip() + "\n", encoding="utf-8")
    return config_path


def _dataset_config(*, screening: dict[str, object] | None = None) -> DatasetConfig:
    payload: dict[str, object] = {
        "scenes": {"horizon_frames": 2, "sample_time": 0.1},
        "task": {"prediction_origin": 1, "prediction_end": 2},
    }
    if screening is not None:
        payload["screening"] = screening
    return DatasetConfig.model_validate(payload)


def test_parse_config_parses_profiles(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [defaults.output]
            precision = "float32"

            [profiles.fast.runtime]
            jobs = 2

            [datasets.demo]
            uses = ["fast"]
            """,
        ),
    )

    assert isinstance(cfg, ProjectConfig)
    assert cfg.defaults is not None
    assert "fast" in cfg.profiles
    assert "demo" in cfg.datasets


def test_parse_config_parses_window(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.scenes]
            horizon_frames = 8
            sample_time = 0.1

            [datasets.demo.task]
            prediction_origin = 3
            prediction_end = 8

            [datasets.demo.scenes.window]
            step = 2
            policy = "partial"
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.scenes.window is not None
    assert resolved.scenes.window.step == 2
    assert resolved.scenes.window.policy == "partial"
    assert resolved.task is not None
    assert resolved.task.prediction_origin == 3
    assert resolved.task.prediction_end == 8


def test_dataset_config_can_clear_inherited_prediction_task(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            task = "none"
            """,
        ),
    )

    assert cfg._resolve_config("demo", _dataset_config()).task is None


def test_inline_prediction_task_must_be_complete(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prediction_end"):
        _ = parse_config(
            _write(
                tmp_path,
                """
                [datasets.demo.task]
                prediction_origin = 1
                """,
            ),
        )


def test_project_defaults_cannot_select_named_dataset_task(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset-specific"):
        _ = parse_config(
            _write(
                tmp_path,
                """
                [defaults]
                task = "benchmark"
                """,
            ),
        )


def test_direct_resolution_requires_named_task_context(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            task = "benchmark"
            """,
        ),
    )

    with pytest.raises(ConfigurationError, match=r"requires.*named_tasks"):
        _ = cfg._resolve_config("demo", _dataset_config())


def test_resolve_applies_defaults_without_dataset(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [defaults.runtime]
            jobs = 8

            [defaults.output]
            trajectory_schema = "canonical"
            precision = "float32"
            recenter_positions = true

            [defaults.output.mds]
            compression = "zstd:3"

            [defaults.output.zarr]
            scene_chunk = 128
            compression_level = 5
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.runtime.jobs == 8
    assert resolved.output.trajectory_schema == "canonical"
    assert resolved.output.precision == "float32"
    assert resolved.output.recenter_positions is True
    assert resolved.output.mds.compression == "zstd:3"
    assert resolved.output.zarr.scene_chunk == 128
    assert resolved.output.zarr.compression_level == 5


def test_resolve_applies_defaults_before_dataset(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [defaults.runtime]
            jobs = 8

            [defaults.output.mds]
            compression = "zstd:3"

            [datasets.demo.runtime]
            jobs = 2
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.runtime.jobs == 2
    assert resolved.output.mds.compression == "zstd:3"


def test_defaults_can_use_profiles(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [profiles.common.runtime]
            jobs = 8

            [defaults]
            uses = ["common"]
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.runtime.jobs == 8


def test_resolve_raises_for_missing_profile(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            uses = ["missing"]
            """,
        ),
    )

    with pytest.raises(ConfigurationError, match="Profile 'missing' not found"):
        _ = cfg._resolve_config(
            "demo",
            DatasetConfig.model_validate({
                "scenes": {"horizon_frames": 2, "sample_time": 0.1},
                "task": {"prediction_origin": 1, "prediction_end": 2},
            }),
        )


def test_resolve_raises_for_missing_defaults_profile(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [defaults]
            uses = ["missing"]
            """,
        ),
    )

    with pytest.raises(ConfigurationError, match="Profile 'missing' not found for defaults"):
        _ = cfg._resolve_config("demo", _dataset_config())


def test_parse_config_rejects_invalid_toml(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        [datasets.demo
        uses = ["fast"]
        """,
    )

    with pytest.raises(ConfigurationError, match="Invalid TOML in config file"):
        _ = parse_config(path)


def test_runtime_override_sets_present_sections() -> None:
    override = config_overrides(
        assign_strategy="scene",
        read_split=None,
        jobs=3,
        trajectory_schema="canonical",
        ratio=(0.7, 0.2, 0.1),
        gap=None,
        segments=None,
    )

    assert override.runtime is not None
    assert override.runtime.jobs == 3
    assert override.output is not None
    assert override.output.trajectory_schema == "canonical"
    assert override.assign is not None

    empty = config_overrides(
        read_strategy=None,
        read_split=None,
        assign_strategy=None,
        jobs=None,
        trajectory_schema=None,
        ratio=None,
        gap=None,
        segments=None,
    )
    assert empty.runtime is None
    assert empty.output is None
    assert empty.read is None
    assert empty.assign is None


def test_runtime_override_requires_native_read_for_splits() -> None:
    with pytest.raises(ConfigurationError, match="read_split"):
        _ = config_overrides(read_split=[DatasetSplit.TRAIN])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ratio": (0.7, 0.2, 0.1)}, "Assignment options require"),
        (
            {"assign_strategy": "none", "ratio": (0.7, 0.2, 0.1)},
            "only valid for scene, source, time, and shuffled-time",
        ),
        ({"assign_strategy": "scene", "gap": 2, "ratio": (0.7, 0.2, 0.1)}, "gap"),
        ({"assign_strategy": "shuffled-time", "ratio": (0.7, 0.2, 0.1)}, "segments"),
    ],
    ids=["strategy-required", "ratio-strategy", "gap-strategy", "segments-required"],
)
def test_runtime_override_rejects_bad_assignment_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        _ = config_overrides(**kwargs)  # pyright: ignore[reportArgumentType]


def test_resolve_disables_inherited_optional_blocks(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            screening = { op = "clear" }

            [datasets.demo.scenes]
            window = "clear"
            resample = "clear"
            lane_change = "clear"
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", inherited_optional_blocks_descriptor())

    assert resolved.screening is None
    assert resolved.scenes.window is None
    assert resolved.scenes.resample is None
    assert resolved.scenes.lane_change is None


def test_screening_extend_is_default(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening.agents.observation_floor]
            rule = "min_observations"
            minimum = 8
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.screening is not None
    assert resolved.screening.cleanup == {}
    assert resolved.screening.scenes == {}
    assert set(resolved.screening.agents) == {"observation_floor"}
    assert isinstance(resolved.screening.agents["observation_floor"], agent.MinObservations)
    assert resolved.screening.agents["observation_floor"].minimum == 8


def test_screening_extend_merges_namespaces(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening.scenes]
            mode = "extend"

            [datasets.demo.screening.agents]
            mode = "extend"

            [datasets.demo.screening.scenes.context_window]
            rule = "scene_window"
            start_frame = 0
            end_frame = 3

            [datasets.demo.screening.agents.observation_floor]
            rule = "min_observations"
            minimum = 8
            """,
        ),
    )

    resolved = cfg._resolve_config(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {"trim_static": {"rule": "exclude", "categories": ["STATIC_OBJECT"]}},
                "scenes": {"min_context": {"rule": "agent_range", "minimum": 2}},
                "agents": {"observation_floor": {"rule": "min_observations", "minimum": 4}},
            },
        ),
    )

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"trim_static"}
    assert set(resolved.screening.scenes) == {"min_context", "context_window"}
    assert set(resolved.screening.agents) == {"observation_floor"}
    assert isinstance(resolved.screening.cleanup["trim_static"], cleanup.ExcludeCategories)
    assert isinstance(resolved.screening.scenes["min_context"], scene.AgentRange)
    assert isinstance(resolved.screening.scenes["context_window"], scene.SceneRequireWindow)
    assert isinstance(resolved.screening.agents["observation_floor"], agent.MinObservations)
    assert resolved.screening.agents["observation_floor"].minimum == 8


def test_screening_replace_discards_inherited(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening.scenes]
            mode = "replace"

            [datasets.demo.screening.agents]
            mode = "replace"

            [datasets.demo.screening.cleanup]
            mode = "replace"

            [datasets.demo.screening.scenes.context_window]
            rule = "scene_window"
            start_frame = 0
            end_frame = 3
            """,
        ),
    )

    resolved = cfg._resolve_config(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {"trim_static": {"rule": "exclude", "categories": ["STATIC_OBJECT"]}},
                "scenes": {"min_context": {"rule": "agent_range", "minimum": 2}},
                "agents": {"observation_floor": {"rule": "min_observations", "minimum": 4}},
            },
        ),
    )

    assert resolved.screening is not None
    assert resolved.screening.cleanup == {}
    assert set(resolved.screening.scenes) == {"context_window"}
    assert resolved.screening.agents == {}
    assert isinstance(resolved.screening.scenes["context_window"], scene.SceneRequireWindow)


def test_screening_remove_drops_names(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening.agents]
            mode = "extend"
            remove = ["shared"]

            [datasets.demo.screening.scenes]
            mode = "extend"
            remove = ["shared"]

            [datasets.demo.screening.cleanup]
            mode = "extend"
            remove = ["shared"]


            """,
        ),
    )

    resolved = cfg._resolve_config(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {
                    "shared": {"rule": "exclude", "categories": ["STATIC_OBJECT"]},
                    "keep_cleanup": {"rule": "exclude", "categories": ["ANIMAL"]},
                },
                "scenes": {
                    "shared": {"rule": "agent_range", "minimum": 2},
                    "keep_scene": {"rule": "scene_frames", "frames": [0]},
                },
                "agents": {
                    "shared": {"rule": "min_observations", "minimum": 4},
                    "keep_agent": {"rule": "min_observations", "minimum": 2},
                },
            },
        ),
    )

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"keep_cleanup"}
    assert set(resolved.screening.scenes) == {"keep_scene"}
    assert set(resolved.screening.agents) == {"keep_agent"}
    assert isinstance(resolved.screening.cleanup["keep_cleanup"], cleanup.ExcludeCategories)
    assert isinstance(resolved.screening.scenes["keep_scene"], scene.SceneRequireFrames)
    assert isinstance(resolved.screening.agents["keep_agent"], agent.MinObservations)


def test_screening_profiles_resolve_before_dataset(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [profiles.base.screening.agents.min_obs]
            rule = "min_observations"
            minimum = 4

            [profiles.base.screening.scenes.min_context]
            rule = "agent_range"
            minimum = 2

            [profiles.strict.screening.agents]
            mode = "extend"

            [profiles.strict.screening.agents.min_obs]
            rule = "min_observations"
            minimum = 8

            [profiles.strict.screening.agents.anchor_present]
            rule = "frames"
            frames = [19]

            [profiles.curated.screening.cleanup]
            mode = "replace"

            [profiles.curated.screening.agents]
            mode = "replace"

            [profiles.curated.screening.scenes]
            mode = "replace"

            [profiles.curated.screening.cleanup.trim_static]
            rule = "exclude"
            categories = ["STATIC_OBJECT", "UNIMPORTANT"]

            [profiles.curated.screening.scenes.category_mix]
            rule = "category_range"
            ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }

            [datasets.demo]
            uses = ["base", "strict", "curated"]

            [datasets.demo.screening.scenes]
            mode = "extend"
            remove = ["category_mix"]

            [datasets.demo.screening.scenes.final_context]
            rule = "agent_range"
            minimum = 3
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"trim_static"}
    assert set(resolved.screening.scenes) == {"final_context"}
    assert isinstance(resolved.screening.cleanup["trim_static"], cleanup.ExcludeCategories)
    assert isinstance(resolved.screening.scenes["final_context"], scene.AgentRange)
    assert resolved.screening.scenes["final_context"].minimum == 3
    assert resolved.screening.agents == {}


def test_map_config_parses_scene_extent(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.map.extraction]
            mode = "scene_extent"
            padding = 1.25
            shape = "bounding_box"

            [datasets.demo.map.edge_types]
            include = ["CURB", "LINE_THIN_DOUBLE"]
            exclude = ["VIRTUAL"]

            [datasets.demo.map.edge_types.remap]
            LINE_THIN_DOUBLE = "LINE_THIN"
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert isinstance(resolved.map.extraction, SceneExtentExtraction)
    assert resolved.map.extraction.padding == pytest.approx(1.25)
    assert resolved.map.extraction.shape == "bounding_box"
    assert resolved.map.edge_types is not None
    assert resolved.map.edge_types.include == frozenset({EdgeType.CURB, EdgeType.LINE_THIN_DOUBLE})
    assert resolved.map.edge_types.exclude == frozenset({EdgeType.VIRTUAL})
    assert resolved.map.edge_types.remap == {EdgeType.LINE_THIN_DOUBLE: EdgeType.LINE_THIN}


def test_map_edge_types_reject_overlap() -> None:
    with pytest.raises(ValueError, match="Conflict"):
        _ = MapEdgeTypeRules.model_validate({
            "include": ["CURB", "VIRTUAL"],
            "exclude": ["VIRTUAL"],
        })


def test_map_edge_types_normalize_conflicts() -> None:
    with pytest.raises(ValueError, match="VIRTUAL"):
        _ = MapEdgeTypeRules.model_validate({"include": ["VIRTUAL"], "exclude": [EdgeType.VIRTUAL]})


def test_map_config_parses_trajectory_buffer(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.map.extraction]
            mode = "trajectory_buffer"
            radius = 6.5
            """,
        ),
    )

    resolved = cfg._resolve_config("demo", _dataset_config())

    assert isinstance(resolved.map.extraction, TrajectoryBufferExtraction)
    assert resolved.map.extraction.radius == pytest.approx(6.5)


def test_agent_rules_load() -> None:
    config = ScreeningConfig.model_validate({
        "agents": {
            "required_frames": {"rule": "frames", "frames": [0, 1, 2]},
            "coverage": {"rule": "window", "start_frame": 0, "end_frame": 2, "min_fraction": 0.5},
            "gap_budget": {
                "rule": "max_gap",
                "maximum": 1,
                "require": {"absolute": 2, "relative": 0.75},
            },
        },
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.agent_rules) == 3
    assert isinstance(compiled.agent_rules[0], agent.AgentRequireFrames)
    assert isinstance(compiled.agent_rules[1], agent.AgentRequireWindow)
    assert isinstance(compiled.agent_rules[2], agent.MaxGap)
    assert compiled.agent_rules[0].rule_id == "required_frames"
    assert compiled.agent_rules[1].rule_id == "coverage"
    assert compiled.agent_rules[2].rule_id == "gap_budget"
    assert compiled.agent_rules[2].require is not None
    assert compiled.agent_rules[2].require.absolute == 2
    assert compiled.agent_rules[2].require.relative == pytest.approx(0.75)


def test_scene_rules_load() -> None:
    config = ScreeningConfig.model_validate({
        "scenes": {
            "agent_bounds": {"rule": "agent_range", "minimum": 1, "maximum": 5},
            "missing_frames": {
                "rule": "max_missing_frames",
                "maximum": 1,
                "selector": {"categories": ["car"]},
            },
        },
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.scene_rules) == 2
    assert isinstance(compiled.scene_rules[0], scene.AgentRange)
    assert compiled.scene_rules[0].rule_id == "agent_bounds"
    assert isinstance(compiled.scene_rules[1], scene.SceneMaxMissingFrames)
    assert compiled.scene_rules[1].rule_id == "missing_frames"


def test_cleanup_rules_load_nested_agent_rules() -> None:
    config = ScreeningConfig.model_validate({
        "cleanup": {
            "keep_only_cars": {"rule": "include", "categories": ["car"]},
            "prune_sparse": {
                "rule": "prune_by",
                "agent_rule": {"rule": "min_observations", "minimum": 3},
            },
        },
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.cleanup_rules) == 2
    assert isinstance(compiled.cleanup_rules[0], cleanup.IncludeCategories)
    assert isinstance(compiled.cleanup_rules[1], cleanup.PruneByRule)
    assert isinstance(compiled.cleanup_rules[1].agent_rule, agent.MinObservations)
    assert compiled.cleanup_rules[0].rule_id == "keep_only_cars"
    assert compiled.cleanup_rules[1].rule_id == "prune_sparse"


def test_prune_by_config_rejects_nested_require() -> None:
    with pytest.raises(ValueError, match="require"):
        ScreeningConfig.model_validate({
            "cleanup": {
                "prune_sparse": {
                    "rule": "prune_by",
                    "agent_rule": {
                        "rule": "min_observations",
                        "minimum": 3,
                        "require": {"absolute": 1},
                    },
                },
            },
        })


def test_require_must_define_threshold() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ScreeningConfig.model_validate({
            "agents": {
                "observation_floor": {"rule": "min_observations", "minimum": 3, "require": {}},
            },
        })


def test_compiled_rules_work_with_screen_data() -> None:
    config = ScreeningConfig.model_validate({
        "cleanup": {"drop_unimportant": {"rule": "exclude", "categories": ["unimportant"]}},
        "scenes": {"enough_agents": {"rule": "agent_range", "minimum": 1}},
        "agents": {"max_missing": {"rule": "max_missing_frames", "maximum": 1}},
    })
    compiled = ScreeningRuleSet.from_config(config)

    frame = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
        "agent_category": [AgentCategory.CAR] * 7,
    })

    screened = screen_data(frame, compiled, columns=TrajectoryColumns())

    assert screened.passes_scene
    assert len(screened.frame) == len(frame)
    assert screened.frame.columns == frame.columns
