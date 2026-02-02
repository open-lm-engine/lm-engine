# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from enum import Enum

import numpy as np
from torch.utils.cpp_extension import load as load_cpp_extension

from ....utils import Communication, ProcessGroupManager, log_rank_0


class Split(Enum):
    train = 0
    valid = 1
    test = 2


_HELPERS = None


def compile_helpers() -> None:
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""

    log_rank_0(logging.INFO, "compiling helpers.cpp")

    build_directory = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_directory, exist_ok=True)

    def _compile():
        global _HELPERS
        _HELPERS = load_cpp_extension(
            "helpers",
            sources=os.path.join(os.path.dirname(__file__), "helpers.cpp"),
            extra_cflags=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC", "-fdiagnostics-color"],
            build_directory=build_directory,
            verbose=True,
        )

    if ProcessGroupManager.get_global_rank() == 0:
        _compile()

    Communication.barrier()

    if ProcessGroupManager.get_global_rank() != 0:
        _compile()


def build_blending_indices(
    dataset_index: np.ndarray, dataset_sample_index: np.ndarray, weights: list[float], num_datasets: int, size: int
) -> None:
    _HELPERS.build_blending_indices(dataset_index, dataset_sample_index, weights, num_datasets, size)


def build_sample_idx(
    sizes: np.ndarray, doc_idx: np.ndarray, sequence_length: int, num_epochs: int, tokens_per_epoch: int
) -> np.ndarray:
    if doc_idx.dtype == np.int32:
        log_rank_0(logging.INFO, f"using int32 for sample idx")
        sample_idx = _HELPERS.build_sample_idx_int32(sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch)
    elif doc_idx.dtype == np.int64:
        log_rank_0(logging.INFO, f"using int64 for sample idx")
        sample_idx = _HELPERS.build_sample_idx_int64(sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch)
    else:
        raise ValueError("unexpected dtype for doc_idx")

    return sample_idx


def normalize(weights: list[float]) -> list[float]:
    """Do non-exponentiated normalization

    Args:
        weights (list[float]): The weights

    Returns:
        list[float]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = (w / w_sum).tolist()
    return w
