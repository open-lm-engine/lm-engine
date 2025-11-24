# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

from __future__ import annotations

import logging
import struct
import time
from functools import lru_cache
from types import TracebackType

import numpy as np

from ...utils import is_multi_storage_client_available, log_rank_0
from .dtype import DType


if is_multi_storage_client_available():
    import msc


_INDEX_HEADER = b"MMIDIDX\x00\x00"


class _IndexWriter:
    """class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (type[np.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[np.number]) -> _IndexWriter:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> _IndexWriter:
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = (msc.open if is_msc(self.idx_path) else open)(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool | None:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (type[BaseException] | None): Exception type

            exc_val (BaseException | None): Exception value

            exc_tb (TracebackType | None): Exception traceback object

        Returns:
            bool | None: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(self, sequence_lengths: list[int], sequence_modes: list[int], document_indices: list[int]) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (list[int]): The length of each sequence

            sequence_modes (list[int]): The mode of each sequences

            document_indices (list[int]): The sequence indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        assert (
            max(sequence_lengths) <= np.iinfo(np.int32).max
        ), "sequence lengths are assumed to be smaller than the max value of np.int32"
        sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = np.array(document_indices, dtype=np.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = np.array(sequence_modes, dtype=np.int8)
            self.idx_writer.write(sequence_modes.tobytes(order="C"))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: list[int]) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (list[int]): The length of each sequence

        Returns:
            list[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += (length if isinstance(length, int) else length.item()) * itemsize
        return list_ptr


class _IndexReader:
    """class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file
        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> _IndexReader:
        log_rank_0(logging.INFO, f"Load the {type(self).__name__} from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        log_rank_0(logging.INFO, f"\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = np.frombuffer(
            self.bin_buffer, dtype=np.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        self.sequence_modes = None
        if multimodal:
            log_rank_0(logging.INFO, f"\tExtract the sequence modes")
            t_beg = time.time()
            self.sequence_modes = np.frombuffer(
                self.bin_buffer,
                dtype=np.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        log_rank_0(logging.INFO, f"> total number of sequences: {len(self)}")
        log_rank_0(logging.INFO, f"> total number of documents: {self.document_indices.shape[0] - 1}")

    def __del__(self) -> None:
        """Clean up the object"""
        self.bin_buffer_mmap._mmap.close()
        del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> tuple[np.int32, np.int64, np.int8 | None]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[np.int64, np.int32, np.int8 | None]: The pointer, length and mode at
            the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


def get_idx_path(path_prefix: str) -> str:
    return f"{path_prefix}.idx"
