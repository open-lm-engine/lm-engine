# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...utils import is_boto3_available, is_multi_storage_client_available
from .dtype import DType


if is_multi_storage_client_available():
    import msc


if is_boto3_available():
    import boto3


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray: ...


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = (msc.open if is_msc(bin_path) else open)(bin_path, mode="rb")
        self._bin_buffer_mmap = np.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        """Read bytes into a np array.

        Args:
            dtype (type[np.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            np.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        return np.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        """Read bytes into a np array.

        Args:
            dtype (Type[np.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            np.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        sequence = np.empty(count, dtype=dtype)
        with (msc.open if is_msc(bin_path) else open)(self._bin_path, mode="rb", buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence


class _S3BinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file from S3

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.

        bin_chunk_nbytes (int, optional): If not None, then maintain an in-memory cache to speed
            up calls to the `read` method. Furthermore, on a cache miss, download this number of
            bytes to refresh the cache. Otherwise (None), do not maintain an in-memory cache.
            A class that inherits from _BinReader may not implement caching in which case it
            should assert that `bin_chunk_nbytes` is None at initialization.
    """

    def __init__(self, bin_path: str, object_storage_config: ObjectStorageConfig) -> None:
        assert object_storage_config.bin_chunk_nbytes > 0
        self._client = boto3.client("s3")
        self._s3_bucket, self._s3_key = parse_s3_path(bin_path)
        self._cache_nbytes = object_storage_config.bin_chunk_nbytes

        self._cache_bytes_start: int
        self._cache_bytes_end: int
        self._cache: bytes | None = None

    def _extract_from_cache(self, offset: int, size: int) -> bytes:
        """Extract `size` bytes starting at `offset` bytes into the cache"""
        assert self._cache is not None
        start = offset - self._cache_bytes_start
        assert start >= 0
        end = start + size
        assert end <= len(self._cache)
        return self._cache[start:end]

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        """Read bytes into a np array.

        Let `size` be the `count` * `DType.size(dtype)`. If the requested span of bytes [`offset`,
        `offset` + `size`) is covered by the in-memory cache maintained by this class, then this
        function extracts the requested span from that cache and returns it. Otherwise, this
        function first refreshes the cache and then extracts the requested span from the refreshed
        cache and returns it.

        The cache is refreshed based on `offset` and `size`. In particular, we divide all the bytes
        in an S3 object into blocks, where each block contains `bin_chunk_nbytes` bytes. We assign
        each block an index starting from 0. We take the block with index (`offset` //
        `bin_chunk_nbytes`) to refresh the cache. If this new block still does not cover the
        requested span, we extend it just enough to include `offset` + `size`.

        Args:
            dtype (Type[np.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            np.ndarray: An array with `count` items and data-type `dtype` constructed from
            reading bytes from the data file starting at `offset`.
        """
        size = count * DType.size(dtype)
        if self._cache is not None and offset >= self._cache_bytes_start and offset + size <= self._cache_bytes_end:
            return np.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)

        bytes_start = (offset // self._cache_nbytes) * self._cache_nbytes
        assert bytes_start >= 0
        assert offset >= bytes_start
        bytes_end = max(bytes_start + self._cache_nbytes, offset + size)
        assert bytes_end >= 1
        self._cache = self._client.get_object(
            Bucket=self._s3_bucket,
            Key=self._s3_key,
            # Subtract 1, because the end of Range is inclusive.
            Range=f"bytes={bytes_start}-{bytes_end - 1}",
        )["Body"].read()
        self._cache_bytes_start = bytes_start
        self._cache_bytes_end = bytes_end
        return np.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)

    def __del__(self) -> None:
        """Clean up the object"""
        self._client.close()


class _MultiStorageClientBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using the multi-storage client.

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str, object_storage_config: ObjectStorageConfig) -> None:
        self._client, self._bin_path = msc.resolve_storage_client(bin_path)

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        size = count * DType.size(dtype)
        buffer = self._client.read(path=self._bin_path, byte_range=msc.types.Range(offset=offset, size=size))
        return np.frombuffer(buffer, dtype=dtype)


# Map of object storage access to the corresponding bin reader
OBJECT_STORAGE_BIN_READERS = {"s3": _S3BinReader, "msc": _MultiStorageClientBinReader}
