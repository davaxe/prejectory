"""Shard decoding plans for Mosaic's standard MDS format."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from struct import unpack_from
from typing import TYPE_CHECKING, Any, cast, final

import numpy as np
from streaming import StreamingDataset
from streaming.base.format.mds.encodings import NDArray, mds_decode
from streaming.base.format.mds.reader import MDSReader as MosaicShardReader
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

# The shape byte packs ndim into its upper six bits and the native-endian
# unsigned dimension width into its lower two. Every possible header is small.
_SHAPE_HEADERS = tuple(f"={ndim}{width}" for ndim in range(64) for width in "BHIQ")


@final
class MDSStreamingDataset(StreamingDataset):
    """Keep Mosaic epoch planning, with synchronous reads of resident local data."""

    _local_iteration: bool = False
    _local_iteration_token: object | None = None

    def enable_local_iteration(self) -> None:
        """Enable the shortcut only when no download, decompression or eviction is needed."""
        self._local_iteration = (
            self.cache_limit is None
            and all(stream.remote is None for stream in self.streams)
            and all(
                isinstance(shard, PlannedMDSShardReader)
                and shard.compression is None
                and Path(shard.dirname, shard.split, shard.raw_data.basename).is_file()
                for shard in cast("list[object]", self.shards)
            )
        )

    @override
    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self._local_iteration:
            yield from super().__iter__()
            return

        # These are the same world/epoch/partition operations as Mosaic's
        # iterator. Only its preparation threads and readiness polling go away.
        self._unique_worker_world = self._unique_rank_world.detect_workers()
        self._parallel_worker_world = self._parallel_rank_world.detect_workers()
        epoch, sample_in_epoch = self._resume_incr_epoch()
        sample_ids = self._get_work(epoch, sample_in_epoch)
        # Starting another epoch invalidates an interrupted iterator, as upstream.
        self._local_iteration_token = token = object()
        for sample_id in sample_ids:
            if token is not self._local_iteration_token:
                return
            if sample_id != -1:
                yield self[sample_id]


@final
class PlannedMDSShardReader(MosaicShardReader):
    """Reuse immutable column codecs and sample offsets for standard MDS shards."""

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
        self._filename = Path(self.dirname, self.split, self.raw_data.basename)
        self._sample_offsets: bytes | None = None

    @override
    def get_sample_data(self, idx: int) -> bytes:
        if not 0 <= idx < self.samples:
            msg = f"Relative sample index {idx} is not present in {self.raw_data.basename}."
            raise IndexError(msg)
        # Offset tables are immutable across download/decompression/eviction.
        # Cache only their four bytes per entry, never rows or open file handles.
        # Opening the file on every access preserves Mosaic's missing-file retry.
        with self._filename.open("rb", buffering=0) as fp:
            offsets = self._sample_offsets
            if offsets is None:
                _ = fp.seek(4)  # Skip the shard's sample count.
                size = 4 * (self.samples + 1)
                offsets = fp.read(size)
                if len(offsets) != size:
                    msg = f"Incomplete sample offset table in {self.raw_data.basename}."
                    raise IndexError(msg)
                self._sample_offsets = offsets
            begin, end = unpack_from("=II", offsets, 4 * idx)
            if end <= begin:
                msg = f"Invalid sample offsets for index {idx} in {self.raw_data.basename}."
                raise IndexError(msg)
            _ = fp.seek(begin)
            data = fp.read(end - begin)
        if len(data) != end - begin:
            msg = f"Incomplete sample at index {idx} in {self.raw_data.basename}."
            raise IndexError(msg)
        return data

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
        return _decode_int
    if encoding == "ndarray" or encoding.startswith("ndarray:"):
        codec = NDArray.from_str(encoding.partition(":")[2])
        if codec.dtype is not None:
            return _FixedDtypeArrayDecoder(np.dtype(codec.dtype), codec.shape)
        return cast("Callable[[bytes], Any]", codec.decode)
    # Custom encodings keep Mosaic's dispatch, including safety validation in
    # StreamingDataset. Do not cache potentially stateful third-party codecs.
    return partial(mds_decode, encoding)


def _decode_int(data: bytes) -> int:
    return unpack_from("=q", data)[0]


@final
class _FixedDtypeArrayDecoder:
    """Parse shapes into Python tuples and view a column's payload without recopying it."""

    def __init__(self, dtype: np.dtype[Any], shape: tuple[int, ...] | None) -> None:
        self._dtype = dtype
        self._shape = shape

    def __call__(self, data: bytes) -> npt.NDArray[Any]:
        offset = 0
        shape = self._shape
        if shape is None:
            header = data[0]
            shape = unpack_from(_SHAPE_HEADERS[header], data, 1)
            offset = 1 + ((header >> 2) << (header & 3))
        # The backing bytes belong to this column, not the entire scene. Output
        # remains read-only like Mosaic, and Torch's default copy stays intact.
        return np.frombuffer(data, self._dtype, offset=offset).reshape(shape)
