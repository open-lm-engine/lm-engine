# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

from __future__ import annotations

import logging
import os
import shutil
from itertools import accumulate

import numpy as np
import torch

from ...utils import cache_file, get_index_cache_path, is_object_storage_path, log_rank_0
from .bin import _MMapBinReader, _MultiStorageClientBinReader
from .dtype import DType
from .idx import _IndexReader, _IndexWriter


class MMapIndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self, path_prefix: str, multimodal: bool = False, cache_path: str | None = None
    ) -> MMapIndexedDataset:
        super().__init__()

        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.cache_path = cache_path

        if is_object_storage_path(path_prefix):
            remote_idx_path = get_idx_path(path_prefix)
            idx_path = get_index_cache_path(remote_idx_path, self.cache_path)
            log_rank_0(logging.INFO, f"downloading {remote_idx_path} to {idx_path}")
            cache_file(remote_idx_path, idx_path)

    def initialize(self, path_prefix: str) -> None:
        is_object_storage = is_object_storage_path(path_prefix)
        idx_path = get_idx_path(path_prefix)

        self.index = _IndexReader(idx_path, self.multimodal)
        self.bin_reader = (_MultiStorageClientBinReader if is_object_storage else _MMapBinReader)(
            get_bin_path(self.path_prefix)
        )

    def __getstate__(self) -> tuple[str, bool]:
        """Get the state during pickling

        Returns:
            tuple[str, bool]: The state tuple
        """
        return self.path_prefix

    def __setstate__(self, state: tuple[str, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (tuple[str, bool]): The state tuple
        """
        path_prefix = state
        self.initialize(path_prefix)

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(self, idx: int | np.integer | slice) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return from the dataset

        Args:
            idx (int | np.integer | slice): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous
            TypeError: When the index is of an unexpected type

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The sequence tokens and
            modes at the index or index slice
        """
        if isinstance(idx, (int, np.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer)
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = np.split(
                self.bin_reader.read(
                    dtype=self.index.dtype, count=sum(sequence_lengths), offset=self.index.sequence_pointers[start]
                ),
                sequence_offsets[:-1],
            )
            return (sequences, sequence_modes) if sequence_modes is not None else sequences
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """

        offset = int(offset)

        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(dtype=self.index.dtype, count=length, offset=sequence_pointer)
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Get the sequence lengths

        Returns:
            np.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """Get the document indices

        Returns:
            np.ndarray: The document indices
        """
        return self.index.document_indices

    @property
    def sequence_modes(self) -> np.ndarray:
        """Get the sequence modes

        Returns:
            np.ndarray: The sequence modes
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

        dtype (np.dtype, optional): The dtype of the index file. Defaults to np.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, bin_path: str, dtype: np.dtype = np.int32, multimodal: bool = False) -> None:
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
        np_array = np.array(tensor.numpy(), dtype=self.dtype)
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
        np_array = np.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.end_document()
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
    return f"{path_prefix}.idx"


def get_bin_path(path_prefix: str) -> str:
    return f"{path_prefix}.bin"
