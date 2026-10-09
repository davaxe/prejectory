# ruff: file-ignore[import-outside-top-level]
# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import pickle  # ruff: ignore[suspicious-pickle-import]
from struct import pack
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


@pytest.mark.parametrize("compression", [None, "zstd:3"])
@pytest.mark.parametrize("all_fixed", [False, True])
def test_planned_mds_decoder_matches_mosaic(
    tmp_path: Path, compression: str | None, *, all_fixed: bool
) -> None:
    streaming = pytest.importorskip("streaming")
    from streaming.base.format.mds.reader import MDSReader as MosaicShardReader

    from prejectory.io.readers._mds import PlannedMDSShardReader
    from prejectory.io.readers.mds import MDSReader

    columns = {
        "integer": "int",
        "dynamic": "ndarray",
        "fixed_dtype": "ndarray:float32",
        "fixed_shape": "ndarray:float64:2,3",
        "text": "str",
        "raw": "bytes",
        "json": "json",
    }
    if all_fixed:
        columns = {name: columns[name] for name in ("integer", "fixed_shape")}
    with streaming.MDSWriter(
        out=str(tmp_path / "unsplit"), columns=columns, compression=compression
    ) as writer:
        for index in range(5):
            row = {
                "integer": index - 2,
                "dynamic": np.arange(300 + index, dtype=np.int64).reshape(1, -1),
                "fixed_dtype": np.arange(1 + index, dtype=np.float32),
                "fixed_shape": np.full((2, 3), index, dtype=np.float64),
                "text": f"row {index}",
                "raw": bytes([index, 0, 255]) if index else b"",
                "json": {"index": index},
            }
            writer.write({name: row[name] for name in columns})
    reader: MDSReader[dict[str, Any]] = MDSReader(path=tmp_path, convert_raw=dict)
    # Access through StreamingDataset also checks compressed shard preparation.
    rows = reader[:]
    index = json.loads((tmp_path / "unsplit" / "index.json").read_text())
    upstream = MosaicShardReader.from_json(str(tmp_path), "unsplit", index["shards"][0])
    planned = PlannedMDSShardReader(upstream)
    _ = planned[0]  # Populate the offset cache before serializing a spawned worker.
    restored = pickle.loads(pickle.dumps(planned))  # ruff: ignore[suspicious-pickle-usage]
    for position, row in enumerate(rows):
        expected = upstream[position]
        restored_row = restored[position]
        assert row.keys() == expected.keys() == restored_row.keys()
        for name, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(row[name], value, strict=True)
                np.testing.assert_array_equal(restored_row[name], value, strict=True)
                assert row[name].flags.writeable == value.flags.writeable
            else:
                assert row[name] == value == restored_row[name]
    assert [row["integer"] for row in reader[[4, 0, 2]]] == [2, -2, 0]
    assert [row["integer"] for row in reader[np.array([3, 1], dtype=np.int64)]] == [1, -1]
    assert reader[-1]["integer"] == 2
    assert [row["integer"] for row in reader] == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("shape_width", range(4))
@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "int8",
        "uint16",
        "int16",
        "float16",
        "uint32",
        "int32",
        "float32",
        "uint64",
        "int64",
        "float64",
    ],
)
def test_fixed_dtype_decoders_match_all_mosaic_shape_widths(dtype: str, shape_width: int) -> None:
    pytest.importorskip("streaming")
    from streaming.base.format.mds.encodings import NDArray

    from prejectory.io.readers._mds import _decoder

    array = np.arange(6, dtype=dtype).reshape(2, 3)
    # MDS writers normally choose the smallest dimension width. Larger legal
    # widths can be covered without allocating arrays with billions of elements.
    header = bytes([(array.ndim << 2) | shape_width])
    data = header + pack("=" + "BHIQ"[shape_width] * array.ndim, *array.shape) + array.tobytes()
    decoded = _decoder(f"ndarray:{dtype}")(data)
    expected = cast("npt.NDArray[Any]", NDArray(dtype).decode(data))
    np.testing.assert_array_equal(decoded, expected, strict=True)
    assert not decoded.flags.writeable


@pytest.mark.parametrize("value", [-(2**63), -1, 0, 2**63 - 1])
def test_planned_integer_decoder_preserves_signed_int64(value: int) -> None:
    pytest.importorskip("streaming")
    from streaming.base.format.mds.encodings import Int

    from prejectory.io.readers._mds import _decoder

    decoded = _decoder("int")(Int().encode(value))
    assert type(decoded) is int
    assert decoded == value


def test_sample_offset_cache_keeps_missing_file_retry(tmp_path: Path) -> None:
    streaming = pytest.importorskip("streaming")
    from prejectory.io.readers.mds import MDSReader

    remote = tmp_path / "remote"
    with streaming.MDSWriter(out=str(remote / "unsplit"), columns={"integer": "int"}) as writer:
        for index in range(3):
            writer.write({"integer": index})
    reader = MDSReader(
        streams=[
            streaming.Stream(remote=str(remote), local=str(tmp_path / "cache"), split="unsplit")
        ],
        convert_raw=dict,
    )
    assert reader[0] == {"integer": 0}
    reader._backend.evict_shard(0)  # ruff: ignore[private-member-access]
    assert reader[2] == {"integer": 2}
    assert [row["integer"] for row in reader] == [0, 1, 2]
