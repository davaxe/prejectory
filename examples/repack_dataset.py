"""Create equivalent MDS or pickle exports from an existing processed dataset.

Example: python examples/repack_dataset.py data/zarr data/waymo-mds --backend mds
The destination must not exist; the source is never modified.
"""

from __future__ import annotations

import argparse
import pickle  # ruff: ignore[suspicious-pickle-import]
from dataclasses import replace
from pathlib import Path

from prejectory.io.dataset import open_dataset
from prejectory.io.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backend", choices=("mds", "pickle"), required=True)
    parser.add_argument("--compression", default=None)
    args = parser.parse_args()
    source = open_dataset(args.source)
    args.destination.mkdir(parents=True, exist_ok=False)
    for split in source.splits:
        records = open_dataset(args.source, split=split)
        destination = args.destination / split
        if args.backend == "mds":
            from streaming import MDSWriter  # ruff: ignore[import-outside-top-level]

            from prejectory.io.encoding.mds import encode_mds_row, mds_columns  # ruff: ignore[import-outside-top-level]

            with MDSWriter(
                out=str(destination),
                columns=mds_columns(source.manifest.precision),
                compression=args.compression,
                size_limit=64 * 1024 * 1024,
            ) as writer:
                for record in records:
                    writer.write(dict(encode_mds_row(record)))
        else:
            destination.mkdir()
            for index, record in enumerate(records):
                with (destination / f"{index:08d}.pkl").open("wb") as handle:
                    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_manifest(args.destination, replace(source.manifest, storage_backend=args.backend))


if __name__ == "__main__":
    main()
