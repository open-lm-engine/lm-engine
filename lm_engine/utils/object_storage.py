# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

MSC_PREFIX = "msc://"


def is_object_storage_path(path: str) -> bool:
    """Ascertain whether a path is in object storage

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in object storage (s3:// or msc://), False otherwise
    """
    return path.startswith(MSC_PREFIX)
