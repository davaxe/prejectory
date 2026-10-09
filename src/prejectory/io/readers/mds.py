"""Framework-neutral readers for Prejectory MDS exports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from typing_extensions import TypedDict, Unpack, override

from prejectory.core.optional import raise_missing_optional_dependency
from prejectory.io.base import DatasetReader, IterableDatasetReader, RecordT, split_directory_name
from prejectory.io.encoding.mds import decode_mds_row

try:
    from streaming import Stream
    from streaming.base.format.mds.reader import MDSReader as MosaicShardReader
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS storage reader", extra="mds")

from prejectory.io.readers._mds import MDSStreamingDataset, PlannedMDSShardReader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

    from prejectory.core.categories import DatasetSplit

__all__ = ["MDSReader", "MDSReaderInitArgs", "Stream"]


class MDSReaderInitArgs(TypedDict, total=False):
    """Optional arguments forwarded to `streaming.StreamingDataset`."""

    download_retry: int
    download_timeout: float
    validate_hash: str
    keep_zip: bool
    epoch_size: int | str | None
    predownload: int
    cache_limit: int | str
    sampling_method: str
    sampling_granularity: int
    partition_algo: str
    num_canonical_nodes: int
    batch_size: int
    shuffle: bool
    shuffle_algo: str
    shuffle_seed: int
    shuffle_block_size: int
    batching_method: str
    allow_unsafe_types: bool
    replication: int
    stream_name: str
    stream_config: dict[str, Any]


class MDSReader(IterableDatasetReader[RecordT], DatasetReader[RecordT]):
    """Read raw scene records from an MDS dataset split.

    Parameters
    ----------
    path : Path, optional
        Path to the local MDS dataset directory. Must contain subdirectories for
        the split to read, e.g. `train` or `unsplit`.
    split : DatasetSplit or str, optional
        Dataset split to read, e.g. `train` or `unsplit`. If not provided, the
        reader will attempt to read from the "unsplit" subdirectory by default.
    streams : Sequence[Stream], optional
        Pre-configured MDS `Stream` objects to read from. If not provided, the
        path and split arguments will be used to construct a `StreamingDataset`.
    convert_raw : Callable[[dict[str, Any]], RecordT], optional
        Function to convert raw MDS rows into the desired output
        format. Only useful when customizing the output format; by default, this
        decodes raw MDS rows into `SceneRecord` objects using the standard
        Prejectory MDS encoding scheme.
    local_iteration : bool, optional
        Use synchronous iteration for fully resident, uncompressed local
        streams without a cache limit (default True). Mosaic still controls
        sampling, shuffling, epoch resumption and worker partitioning. Set False
        to use its background shard preparation for every stream.
    reader_args : MDSReaderInitArgs, optional
        Additional keyword arguments forwarded to the `StreamingDataset`
        constructor. `batch_size` defaults to 1 for direct iteration; set it to
        match your DataLoader when training. `predownload` defaults to at least
        64 samples to keep background shard preparation ahead of consumption.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        split: DatasetSplit | str | None = None,
        streams: Sequence[Stream] | None = None,
        convert_raw: Callable[[Mapping[str, Any]], RecordT] = decode_mds_row,
        local_iteration: bool = True,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        super().__init__()
        batch_size = reader_args.setdefault("batch_size", 1)
        # A batch size of one otherwise shrinks Mosaic's default window to
        # eight samples, repeatedly starving its background preparation thread.
        _ = reader_args.setdefault("predownload", max(64, 8 * batch_size))
        self._convert_record: Callable[[Mapping[str, Any]], RecordT] = convert_raw
        if path is not None and streams is not None:
            msg = "Provide either path or streams, not both."
            raise ValueError(msg)
        if path is None and streams is None:
            msg = "Either `path` or `streams` must be provided."
            raise ValueError(msg)

        if path is not None:
            self._backend: MDSStreamingDataset = MDSStreamingDataset(
                local=Path(path).as_posix(),
                split=split_directory_name(split),
                **reader_args,
            )
        else:
            self._backend = MDSStreamingDataset(streams=streams, **reader_args)

        # Replace only standard MDS shards, after Mosaic validates their codecs.
        # Downloads, eviction and epoch partitioning remain Mosaic's job.
        for index, shard in enumerate(cast("list[object]", self._backend.shards)):
            if type(shard) is MosaicShardReader:
                self._backend.shards[index] = PlannedMDSShardReader(shard)
        if local_iteration:
            self._backend.enable_local_iteration()

    @override
    def __len__(self) -> int:
        return len(self._backend)

    @override
    def __iter__(self) -> Iterator[RecordT]:
        yield from (self._convert_record(record) for record in self._backend)

    @overload
    def __getitem__(self, at: int) -> RecordT: ...

    @overload
    def __getitem__(self, at: list[int] | npt.NDArray[np.int64] | slice) -> list[RecordT]: ...

    @override
    def __getitem__(
        self,
        at: int | slice | list[int] | npt.NDArray[np.int64],
    ) -> RecordT | list[RecordT]:
        """Return one or more decoded scene records.

        This will implicitly raise error if the specified index or indices are
        out of bounds.

        Parameters
        ----------
        at : int or slice or list of int
            Index or indices of the scene record(s) to return.

        Returns
        -------
        RecordT or list of RecordT
            The decoded scene record(s) at the specified index or indices.

        """
        out: Mapping[str, Any] | list[Mapping[str, Any]] = self._backend[at]
        if isinstance(out, list):
            return [self._convert_record(record) for record in out]
        return self._convert_record(out)
