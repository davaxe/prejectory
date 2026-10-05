# ruff: file-ignore[import-outside-top-level]
from __future__ import annotations

import json
import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


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
