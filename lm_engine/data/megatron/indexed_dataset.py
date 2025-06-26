# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

from __future__ import annotations

import logging
import os
import shutil
import struct
import time
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType

import numpy
import torch

from ...utils import log_rank_0


_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the MMapIndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: int | type[numpy.number]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (int | type[numpy.number]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: int | None) -> type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (int | None): The number of elements to be indexed

        Returns:
            type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexWriter:
    """class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> _IndexWriter:
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
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
            max(sequence_lengths) <= numpy.iinfo(numpy.int32).max
        ), "sequence lengths are assumed to be smaller than the max value of np.int32"
        sequence_lengths = numpy.array(sequence_lengths, dtype=numpy.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = numpy.array(sequence_pointers, dtype=numpy.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = numpy.array(document_indices, dtype=numpy.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = numpy.array(sequence_modes, dtype=numpy.int8)
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

    def __init__(self, idx_path: str, multimodal: bool) -> None:
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

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        log_rank_0(logging.INFO, f"\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        self.sequence_modes = None
        if multimodal:
            log_rank_0(logging.INFO, f"\tExtract the sequence modes")
            t_beg = time.time()
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
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
    def __getitem__(self, idx: int) -> tuple[numpy.int32, numpy.int64, numpy.int8 | None]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int64, numpy.int32, numpy.int8 | None]: The pointer, length and mode at
            the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class MMapIndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, path_prefix: str, multimodal: bool = False) -> None:
        super().__init__()
        self.path_prefix = None
        self.multimodal = None

        self.index = None
        self.bin_buffer = None
        self.bin_buffer_mmap = None

        self.initialize(path_prefix, multimodal)

    def initialize(self, path_prefix: str, multimodal: bool) -> None:
        """Initialize the dataset

        This method is called by MMapIndexedDataset.__init__ during object creation and by
        MMapIndexedDataset.__setstate__ during un-puckling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal
        """
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.index = _IndexReader(get_idx_path(self.path_prefix), self.multimodal)
        self.bin_buffer_mmap = numpy.memmap(get_bin_path(self.path_prefix), mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

    def __getstate__(self) -> tuple[str, bool]:
        """Get the state during pickling

        Returns:
            tuple[str, bool]: The state tuple
        """
        return self.path_prefix, self.multimodal

    def __setstate__(self, state: tuple[str, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (tuple[str, bool]): The state tuple
        """
        path_prefix, multimodal = state
        self.initialize(path_prefix, multimodal)

    def __del__(self) -> None:
        """Clean up the object"""
        if self.bin_buffer_mmap is not None:
            self.bin_buffer_mmap._mmap.close()
        del self.bin_buffer_mmap
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(self, idx: int | numpy.integer | slice) -> numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]:
        """Return from the dataset

        Args:
            idx (int | numpy.integer | slice): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous
            TypeError: When the index is of an unexpected type

        Returns:
            numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]: The sequence tokens and
            modes at the index or index slice
        """
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = numpy.frombuffer(
                self.bin_buffer,
                dtype=self.index.dtype,
                count=sequence_length,
                offset=sequence_pointer,
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                numpy.frombuffer(
                    self.bin_buffer,
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> numpy.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """

        offset = int(offset)

        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = numpy.frombuffer(self.bin_buffer, dtype=self.index.dtype, count=length, offset=sequence_pointer)
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def get_document_indices(self) -> numpy.ndarray:
        """Get the document indices

        This method is slated for deprecation.

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def set_document_indices(self, document_indices: numpy.ndarray) -> None:
        """Set the document indices

        This method is slated for deprecation.

        Args:
            document_indices (numpy.ndarray): The document indices
        """
        self.index.document_indices = document_indices

    @property
    def sequence_modes(self) -> numpy.ndarray:
        """Get the sequence modes

        Returns:
            numpy.ndarray: The sequence modes
        """
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the MMapIndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the MMapIndexedDataset exists on disk at the prefix
        """
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(get_bin_path(path_prefix))


class MMapIndexedDatasetBuilder:
    """Builder class for the MMapIndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (numpy.dtype, optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, bin_path: str, dtype: numpy.dtype = numpy.int32, multimodal: bool = False) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(self, tensor: torch.Tensor, lengths: list[int], modes: list[int] | None = None) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add
            lengths (List[int]): The lengths of each item in the document
            modes (Optional[List[int]], optional): The modes for each item in the document.
            Defaults to None.
        """
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def end_document(self) -> None:
        """Finalize the document, for use with MMapIndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix: str) -> None:
        """Add an entire MMapIndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), multimodal=self.multimodal)
        assert index.dtype == self.dtype

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        if self.multimodal:
            self.sequence_modes.extend(index.sequence_modes)

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the data file
    """
    return path_prefix + ".bin"
