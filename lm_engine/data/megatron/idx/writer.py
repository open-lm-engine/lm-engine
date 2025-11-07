# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

from __future__ import annotations

import struct
from types import TracebackType

import numpy as np

from ..dtype import DType
from .reader import INDEX_HEADER


class IndexWriter:
    """class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file
        dtype (type[np.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[np.number]) -> IndexWriter:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> IndexWriter:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            self.idx_writer = msc.open(self.idx_path, "wb")
        else:
            self.idx_writer = open(self.idx_path, "wb")

        # fixed, vestigial practice
        self.idx_writer.write(INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
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
        self.idx_writer.write(np.array(sequence_lengths, dtype=np.int32).tobytes(order="C"))

        # the byte offsets for all sequences
        self.idx_writer.write(np.array(sequence_pointers, dtype=np.int64).tobytes(order="C"))

        # the sequence indices marking the end of each document
        self.idx_writer.write(np.array(document_indices, dtype=np.int64).tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            self.idx_writer.write(np.array(sequence_modes, dtype=np.int8).tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: list[int]) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (list[int]): The length of each sequence

        Returns:
            list[int]: The pointer to the beginning of each sequence
        """
        itemsize = np.int64(DType.size(self.dtype))
        curr_ptr = np.int64(0)
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr.item())
            curr_ptr += length * itemsize
        return list_ptr
