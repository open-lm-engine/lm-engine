# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Any, Dict, Protocol

from .packages import is_multi_storage_client_available
from .parallel import ProcessGroupManager


if is_multi_storage_client_available():
    import multistorageclient as msc

MSC_PREFIX = "msc://"


class S3Client(Protocol):
    """The protocol which all s3 clients should abide by"""

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        """Download the file from S3 to the local file system"""
        ...

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        """Upload the file to S3"""
        ...

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """Get the metadata of the file in S3"""
        ...

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]:
        """Get the file from S3"""
        ...

    def close(self) -> None:
        """Close the S3 client"""
        ...


def is_object_storage_path(path: str) -> bool:
    """Ascertain whether a path is in object storage

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in object storage (s3:// or msc://), False otherwise
    """
    return path.startswith(MSC_PREFIX)


def get_index_cache_path(idx_path: str, path_to_idx_cache: str) -> str:
    """Get the index cache path for the given path

    Args:
        idx_path (str): The path to the index file
        path_to_idx_cache (str): path to the idx cache

    Returns:
        str: The index cache path
    """
    if is_object_storage_path(idx_path):
        return os.path.join(path_to_idx_cache, idx_path.removeprefix(MSC_PREFIX))

    raise ValueError(f"Invalid path: {idx_path}")


def cache_file(remote_path: str, local_path: str) -> None:
    """Download a file from object storage to a local path with distributed training support.
    The download only happens on Rank 0, and other ranks will wait for the file to be available.

    Note that this function does not include any barrier synchronization. The caller (typically
    in blended_megatron_dataset_builder.py) is responsible for ensuring proper synchronization
    between ranks using torch.distributed.barrier() after this function returns.

    Args:
        remote_path (str): The URL of the file to download (e.g., s3://bucket/path/file.idx
            or msc://profile/path/file.idx)
        local_path (str): The local destination path where the file should be saved

    Raises:
        ValueError: If the remote_path is not a valid S3 or MSC path
    """

    if is_object_storage_path(remote_path):
        if not ProcessGroupManager.is_initialized() or ProcessGroupManager.get_global_rank() == 0:
            msc.download_file(remote_path, local_path)

        assert os.path.exists(local_path)
    else:
        raise ValueError(f"Invalid path: {remote_path}")
