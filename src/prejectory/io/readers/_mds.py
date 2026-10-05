"""Shard decoding plans for Mosaic's standard MDS format."""

from __future__ import annotations

from functools import partial
from struct import unpack_from
from typing import TYPE_CHECKING, Any, cast, final

from streaming.base.format.mds.encodings import Int, NDArray, mds_decode
from streaming.base.format.mds.reader import MDSReader as MosaicShardReader
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class PlannedMDSShardReader(MosaicShardReader):
    """Reuse immutable column codecs while retaining Mosaic shard I/O."""

    def __init__(self, reader: MosaicShardReader) -> None:
        """Build a decoding plan from an already validated Mosaic shard."""
        super().__init__(
            dirname=reader.dirname,
            split=reader.split,
            column_encodings=reader.column_encodings,
            column_names=reader.column_names,
            column_sizes=reader.column_sizes,
            compression=reader.compression,
            hashes=reader.hashes,
            raw_data=reader.raw_data,
            samples=reader.samples,
            size_limit=reader.size_limit,
            zip_data=reader.zip_data,
        )
        self._decoders = tuple(_decoder(encoding) for encoding in reader.column_encodings)
        # MDS stores one native-endian uint32 size for each variable column.
        variable_columns = sum(not size for size in self.column_sizes)
        # Keep the format string (not an unpicklable Struct) for spawned workers.
        self._size_header = "=" + "I" * variable_columns
        self._size_header_bytes = 4 * variable_columns

    @override
    def decode_sample(self, data: bytes) -> dict[str, Any]:
        sizes = iter(unpack_from(self._size_header, data))
        offset = self._size_header_bytes
        sample: dict[str, Any] = {}
        for name, decode, fixed_size in zip(
            self.column_names, self._decoders, self.column_sizes, strict=True
        ):
            size = fixed_size or next(sizes)
            sample[name] = decode(data[offset : offset + size])
            offset += size
        return sample


def _decoder(encoding: str) -> Callable[[bytes], Any]:
    if encoding == "int":
        return Int().decode
    if encoding == "ndarray" or encoding.startswith("ndarray:"):
        return cast("Callable[[bytes], Any]", NDArray.from_str(encoding.partition(":")[2]).decode)
    # Custom encodings keep Mosaic's dispatch, including safety validation in
    # StreamingDataset. Do not cache potentially stateful third-party codecs.
    return partial(mds_decode, encoding)
