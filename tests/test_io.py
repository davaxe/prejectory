# ruff: file-ignore[import-outside-top-level]
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from prejectory.core.errors import ManifestCompatibilityError
from prejectory.io import DatasetManifest, PredictionBounds, PredictionTaskManifest, read_manifest
from prejectory.io.backends.pickle import PickleWriter
from prejectory.io.encoding import encode_scene_record
from prejectory.io.encoding.mds import decode_mds_row, encode_mds_row
from prejectory.io.manifest import write_manifest
from prejectory.io.readers import PickleReader
from tests.support import assert_scene_record_equal, output_config

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from prejectory.core.scene import Scene
    from prejectory.io.records import SceneRecord


@dataclass(slots=True)
class CustomPickleRecord:
    scene_number: int
    dataset: object
    values: npt.NDArray[Any]
    source: str


def test_split_scene_record_rejects_bad_bounds(scene: Scene) -> None:
    record = encode_scene_record(scene, dtype=np.float64)

    with pytest.raises(ValueError, match="Prediction bounds"):
        _ = record.forecast(PredictionBounds(record.horizon_frames + 1, record.horizon_frames + 2))


def test_encode_scene_record_uses_passed_ids(scene: Scene) -> None:
    scene = replace(scene, passed_agent_ids=frozenset({10}))
    record = encode_scene_record(scene, dtype=np.float32)

    np.testing.assert_array_equal(record.screened_agent_mask, np.array([True, False]))
    np.testing.assert_array_equal(record.agent_ids, np.array([10, 20], dtype=np.int64))


def test_pickle_writer_roundtrip(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(
        output_dir=output_dir,
        config=output_config(),
        prediction_bounds=PredictionBounds(2, 3),
        splits=None,
    )

    expected = encode_scene_record(
        scene,
        dtype=np.float32,
        prediction_bounds=PredictionBounds(2, 3),
    )
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir)
    assert len(reader) == 1
    assert_scene_record_equal(reader[0], expected)


def test_pickle_writer_accepts_record_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(record: SceneRecord) -> CustomPickleRecord:
        return CustomPickleRecord(
            scene_number=record.scene_number,
            dataset=record.dataset_id,
            values=record.features[:, :1, 0],
            source="record",
        )

    writer = PickleWriter(
        output_dir=output_dir,
        config=output_config(),
        splits=None,
        record_transform=transform,
    )
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir, record_type=CustomPickleRecord)
    record = reader[0]

    assert record.scene_number == scene.scene_number
    assert record.dataset is None
    assert record.source == "record"
    assert record.values.shape == (2, 1)


def test_pickle_writer_accepts_scene_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(scene: Scene) -> CustomPickleRecord:
        return CustomPickleRecord(
            scene_number=scene.scene_number,
            dataset=scene.dataset,
            values=np.array([scene.horizon_frames], dtype=np.int32),
            source="scene",
        )

    writer = PickleWriter(
        output_dir=output_dir,
        config=output_config(),
        splits=None,
        scene_transform=transform,
    )
    writer.write(scene)
    writer.finish_local()

    record = PickleReader(output_dir, record_type=CustomPickleRecord)[0]

    assert record.scene_number == scene.scene_number
    assert record.dataset == "demo"
    assert record.source == "scene"
    np.testing.assert_array_equal(record.values, np.array([scene.horizon_frames], dtype=np.int32))


def test_pickle_writer_rejects_multiple_transforms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record_transform"):
        _ = PickleWriter(
            output_dir=tmp_path,
            config=output_config(),
            splits=None,
            record_transform=lambda record: record,
            scene_transform=lambda scene: scene,
        )


def test_mds_writer_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("streaming")
    scene = replace(scene, dataset="demo")

    from prejectory.io.backends.mds import MDSDatasetWriter
    from prejectory.io.readers import MDSReader

    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir=output_dir,
        config=output_config(),
        splits=None,
        parallel=False,
    )

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    reader = MDSReader(path=output_dir)
    assert len(reader) == 1
    assert reader[0].ego_agent_id == 10
    assert_scene_record_equal(reader[0], expected)


def test_zarr_writer_roundtrip_across_worker_shards(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("zarr")
    from prejectory.io.backends.zarr import ZarrDatasetWriter
    from prejectory.io.readers import ZarrReader

    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "zarr"
    bounds = PredictionBounds(2, 3)
    for worker_id, scene_number in enumerate((7, 8)):
        writer = ZarrDatasetWriter(
            output_dir=output_dir,
            identifier=worker_id,
            config=output_config(),
            prediction_bounds=bounds,
        )
        writer.write(replace(scene, scene_number=scene_number))
        writer.finish_local()

    reader = ZarrReader(output_dir)
    assert len(reader) == 2
    assert reader[-1].scene_number == 8
    expected = encode_scene_record(scene, dtype=np.float32, prediction_bounds=bounds)
    assert_scene_record_equal(reader[0], expected)


def test_zarr_writer_flushes_without_closing_shard(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("zarr")
    from prejectory.io.backends.zarr import ZarrDatasetWriter
    from prejectory.io.readers import ZarrReader

    writer = ZarrDatasetWriter(tmp_path, identifier=0, config=output_config())
    writer.write(replace(scene, scene_number=1))
    writer.flush_local()
    assert [record.scene_number for record in ZarrReader(tmp_path)] == [1]

    writer.write(replace(scene, scene_number=2))
    writer.flush_local()
    writer.finish_local()
    ZarrDatasetWriter.finish_dataset(tmp_path, splits=None)
    assert [record.scene_number for record in ZarrReader(tmp_path)] == [1, 2]


def test_zarr_writer_preserves_empty_map_shapes(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("zarr")
    from prejectory.io.backends.zarr import ZarrDatasetWriter
    from prejectory.io.readers import ZarrReader

    writer = ZarrDatasetWriter(tmp_path, identifier=0, config=output_config())
    writer.write(replace(scene, map_key=None, map_resolver=None))
    writer.finish_local()

    record = ZarrReader(tmp_path)[0]
    assert record.map_node_positions.shape == (0, 2)
    assert record.map_edge_indices.shape == (2, 0)


def test_mds_reader_combines_streams_with_per_row_prediction_bounds(
    tmp_path: Path,
    scene: Scene,
) -> None:
    pytest.importorskip("streaming")
    from streaming import Stream

    from prejectory.io.backends.mds import MDSDatasetWriter
    from prejectory.io.readers import MDSReader

    roots: list[Path] = []
    for index, bounds in enumerate((PredictionBounds(1, 3), PredictionBounds(2, 3))):
        root = tmp_path / f"stream-{index}"
        writer = MDSDatasetWriter(
            output_dir=root,
            config=output_config(),
            prediction_bounds=bounds,
            splits=None,
            parallel=False,
        )
        writer.write(replace(scene, scene_number=index))
        writer.finish_local()
        roots.append(root)

    reader = MDSReader(
        streams=[Stream(local=str(root), split="unsplit") for root in roots],
        shuffle=True,
        batch_size=1,
    )
    records = list(reader)

    assert {(record.prediction_origin, record.prediction_end) for record in records} == {
        (1, 3),
        (2, 3),
    }
    assert {
        (record.forecast().observation_length, record.forecast().future_length)
        for record in records
    } == {(1, 2), (2, 1)}


def test_mds_writer_accepts_transform_with_columns(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("streaming")
    scene = replace(scene, dataset="demo")

    from prejectory.io.backends.mds import MDSDatasetWriter
    from prejectory.io.readers import MDSReader

    def transform(record: SceneRecord) -> dict[str, Any]:
        observation_length = 1
        return {
            "scene_number": int(record.scene_number),
            "history": record.features[:, :observation_length],
            "future": record.features[:, observation_length:],
            "history_mask": record.valid_mask[:, :observation_length].astype(np.uint8, copy=False),
            "future_mask": record.valid_mask[:, observation_length:].astype(np.uint8, copy=False),
        }

    columns = {
        "scene_number": "int",
        "history": "ndarray:float32",
        "future": "ndarray:float32",
        "history_mask": "ndarray:uint8",
        "future_mask": "ndarray:uint8",
    }
    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir=output_dir,
        config=output_config(),
        splits=None,
        parallel=False,
        record_transform=transform,
        mds_columns=columns,
    )
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    raw = MDSReader(path=output_dir, convert_raw=lambda record: record)[0]
    expected = encode_scene_record(scene, dtype=np.float32)

    assert int(raw["scene_number"]) == scene.scene_number
    np.testing.assert_allclose(raw["history"], expected.features[:, :1])
    np.testing.assert_allclose(raw["future"], expected.features[:, 1:])
    np.testing.assert_array_equal(raw["history_mask"], expected.valid_mask[:, :1].astype(np.uint8))
    np.testing.assert_array_equal(raw["future_mask"], expected.valid_mask[:, 1:].astype(np.uint8))


def test_mds_writer_requires_columns_for_custom_transform(tmp_path: Path) -> None:
    pytest.importorskip("streaming")

    from prejectory.io.backends.mds import MDSDatasetWriter

    with pytest.raises(ValueError, match="mds_columns"):
        _ = MDSDatasetWriter(
            output_dir=tmp_path,
            config=output_config(),
            splits=None,
            parallel=False,
            record_transform=lambda record: {"scene_number": record.scene_number},
        )


def test_mds_encoder_decoder_roundtrip(scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    expected = encode_scene_record(
        scene,
        dtype=np.float32,
        prediction_bounds=PredictionBounds(2, 3),
    )
    record = encode_mds_row(expected)
    decoded = decode_mds_row(record)

    assert_scene_record_equal(decoded, expected)


def test_split_scene_record_preserves_identity(scene: Scene) -> None:
    record = encode_scene_record(scene, dtype=np.float32)

    split = record.forecast(PredictionBounds(2, int(record.features.shape[1])))

    np.testing.assert_array_equal(split.agent_ids, record.agent_ids)
    assert split.ego_agent_id == record.ego_agent_id


def test_manifest_write_and_read_roundtrip(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset="test_dataset",
        storage_backend="pickle",
        prejectory_version="2.0.0",
        source_trajectory_schema="positions_only",
        source_trajectory_schema_fields=("frame", "id", "x", "y", "agent_category"),
        trajectory_schema="canonical",
        trajectory_schema_fields=(
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
        ),
        derived_features=("vx", "vy", "yaw"),
        feature_columns=("x", "y", "vx", "vy", "ax", "ay", "yaw"),
        horizon_frames=10,
        precision="float32",
        recenter_positions=True,
        has_map=True,
        sample_time=0.1,
        original_sample_time=0.1,
        prediction_task=PredictionTaskManifest(
            name="benchmark",
            source_prediction_origin=4,
            source_prediction_end=10,
            prediction_origin=4,
            prediction_end=10,
        ),
    )

    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)

    assert loaded == manifest
    assert loaded.dataset_names == ("test_dataset",)


def test_manifest_rejects_legacy_pre_rename_format() -> None:
    with pytest.raises(ManifestCompatibilityError, match=r"version '1'.*Supported version: 3"):
        _ = DatasetManifest.from_json_dict({"format_version": 1})


def test_manifest_rejects_bad_prediction_bounds() -> None:
    with pytest.raises(ValueError, match="effective prediction bounds"):
        _ = DatasetManifest(
            dataset="test_dataset",
            storage_backend="pickle",
            prejectory_version="2.0.0",
            source_trajectory_schema="positions_only",
            source_trajectory_schema_fields=("frame", "id", "x", "y", "agent_category"),
            trajectory_schema="canonical",
            trajectory_schema_fields=(
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
            ),
            derived_features=("vx", "vy", "yaw"),
            feature_columns=("x", "y", "vx", "vy", "ax", "ay", "yaw"),
            horizon_frames=10,
            precision="float32",
            recenter_positions=True,
            has_map=True,
            sample_time=0.1,
            original_sample_time=0.1,
            prediction_task=PredictionTaskManifest(
                name=None,
                source_prediction_origin=4,
                source_prediction_end=11,
                prediction_origin=4,
                prediction_end=11,
            ),
        )


def _build_pickle_reader(
    tmp_path: Path,
    scene: Scene,
    *,
    bounds: PredictionBounds | None = None,
) -> tuple[PickleReader, SceneRecord]:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(
        output_dir=output_dir,
        config=output_config(),
        prediction_bounds=bounds,
        splits=None,
    )

    expected = encode_scene_record(scene, dtype=np.float32, prediction_bounds=bounds)
    writer.write(scene)
    writer.finish_local()

    return PickleReader(output_dir), expected


def _to_numpy(tensor: torch.Tensor) -> npt.NDArray[Any]:
    return cast("npt.NDArray[Any]", tensor.detach().cpu().numpy())


def _assert_tensor_allclose(tensor: torch.Tensor, expected: npt.NDArray[Any]) -> None:
    np.testing.assert_allclose(_to_numpy(tensor), expected)


def _assert_tensor_array_equal(tensor: torch.Tensor, expected: npt.NDArray[Any]) -> None:
    np.testing.assert_array_equal(_to_numpy(tensor), expected)


def test_torch_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from prejectory.io.adapters.torch import TorchSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    dataset = TorchSceneDataset(reader)
    record = dataset[0]
    assert next(iter(dataset)).scene_number == record.scene_number

    assert record.scene_number == expected.scene_number
    assert record.dataset_id == expected.dataset_id
    _assert_tensor_allclose(record.position_offset, expected.position_offset)
    _assert_tensor_array_equal(record.agent_ids, expected.agent_ids)
    _assert_tensor_array_equal(record.agent_types, expected.agent_types)
    assert record.ego_agent_id == expected.ego_agent_id
    _assert_tensor_array_equal(record.screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(record.features, expected.features)
    _assert_tensor_array_equal(record.valid_mask, expected.valid_mask)
    _assert_tensor_allclose(record.map_node_positions, expected.map_node_positions)
    _assert_tensor_array_equal(record.map_edge_indices, expected.map_edge_indices)
    _assert_tensor_array_equal(record.map_node_types, expected.map_node_types)
    _assert_tensor_array_equal(record.map_edge_types, expected.map_edge_types)


def test_torch_scene_record_splits_features(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from prejectory.io.adapters.torch import TorchSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    record = TorchSceneDataset(reader)[0]
    split = record.forecast(PredictionBounds(2, int(record.features.shape[1])))

    assert split.scene_number == expected.scene_number
    assert split.dataset_id == expected.dataset_id
    _assert_tensor_allclose(split.position_offset, expected.position_offset)
    _assert_tensor_array_equal(split.agent_ids, expected.agent_ids)
    assert split.ego_agent_id == expected.ego_agent_id
    _assert_tensor_allclose(split.history_features, expected.features[:, :2])
    _assert_tensor_array_equal(split.history_mask, expected.valid_mask[:, :2])
    _assert_tensor_allclose(split.future_features, expected.features[:, 2:])
    _assert_tensor_array_equal(split.future_mask, expected.valid_mask[:, 2:])
    _assert_tensor_array_equal(split.map_edge_indices, expected.map_edge_indices)


def test_torch_forecast_dataset_uses_row_bounds_and_explicit_override(
    tmp_path: Path,
    scene: Scene,
) -> None:
    pytest.importorskip("torch")
    from prejectory.io.adapters.torch import TorchForecastDataset

    reader, expected = _build_pickle_reader(tmp_path, scene, bounds=PredictionBounds(2, 3))
    split = TorchForecastDataset(reader)[0]
    _assert_tensor_allclose(split.history_features, expected.features[:, :2])
    _assert_tensor_allclose(split.future_features, expected.features[:, 2:3])

    overridden = TorchForecastDataset(reader, bounds=PredictionBounds(1, 2))[0]
    _assert_tensor_allclose(overridden.history_features, expected.features[:, :1])
    _assert_tensor_allclose(overridden.future_features, expected.features[:, 1:2])


def test_torch_forecast_dataset_rejects_task_free_record(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from prejectory.io.adapters.torch import TorchForecastDataset

    reader, _ = _build_pickle_reader(tmp_path, scene)
    with pytest.raises(ValueError, match="no prediction bounds"):
        _ = TorchForecastDataset(reader)[0]


def test_pyg_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from prejectory.io.adapters.pyg import HeteroSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    dataset = HeteroSceneDataset(reader)
    record = dataset.get(0)
    assert next(iter(dataset)).scene_number == record.scene_number

    assert record.scene_number == expected.scene_number
    assert record.dataset_id == (-1 if expected.dataset_id is None else expected.dataset_id)
    _assert_tensor_allclose(record.position_offset, expected.position_offset)
    _assert_tensor_allclose(record["agent"].features, expected.features)
    _assert_tensor_array_equal(record["agent"].agent_id, expected.agent_ids)
    _assert_tensor_array_equal(record["agent"].valid_mask, expected.valid_mask)
    _assert_tensor_array_equal(record["agent"].agent_type, expected.agent_types)
    _assert_tensor_array_equal(record["agent"].screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(record["map"].x, expected.map_node_positions)
    _assert_tensor_array_equal(record["map"].node_type, expected.map_node_types)
    _assert_tensor_array_equal(
        record["map", "connects", "map"].edge_index,
        expected.map_edge_indices,
    )
    _assert_tensor_array_equal(record["map", "connects", "map"].edge_type, expected.map_edge_types)
    assert record.ego_agent_id == expected.ego_agent_id


def test_pyg_collate_pads_full_horizon(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from prejectory.io.adapters.pyg import HeteroSceneDataset, collate_hetero_with_time_padding

    reader, _ = _build_pickle_reader(tmp_path, scene)
    record = HeteroSceneDataset(reader).get(0)

    shorter = record.clone()
    shorter["agent"].features = shorter["agent"].features[:, :1, :]
    shorter["agent"].valid_mask = shorter["agent"].valid_mask[:, :1]

    batch = collate_hetero_with_time_padding([shorter, record])

    assert int(batch["agent"].features.size(1)) == int(record["agent"].features.size(1))


def test_pyg_forecast_collate_aligns_history_and_future(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from prejectory.io.adapters.pyg import (
        HeteroForecastDataset,
        collate_forecast_hetero_with_time_padding,
    )

    reader, _ = _build_pickle_reader(tmp_path, scene)
    short_history = HeteroForecastDataset(reader, bounds=PredictionBounds(1, 3)).get(0)
    long_history = HeteroForecastDataset(reader, bounds=PredictionBounds(2, 3)).get(0)
    batch = collate_forecast_hetero_with_time_padding([short_history, long_history])

    assert tuple(batch["agent"].history_features.shape[1:]) == (2, 7)
    assert tuple(batch["agent"].future_features.shape[1:]) == (2, 7)
    assert not bool(batch["agent"].history_mask[:2, 0].any())
    assert not bool(batch["agent"].future_mask[2:, 1].any())
