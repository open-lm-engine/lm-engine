# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from enum import Enum

import numpy
import torch
from torch.utils.cpp_extension import load as load_cpp_extension

from ....utils import ProcessGroupManager, log_rank_0


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers() -> None:
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""

    log_rank_0(logging.INFO, "compiling helpers.cpp")

    build_directory = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_directory, exist_ok=True)

    if ProcessGroupManager.get_global_rank() == 0:
        load_cpp_extension(
            "helpers",
            sources=os.path.join(os.path.dirname(__file__), "helpers.cpp"),
            extra_cflags=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC", "-fdiagnostics-color"],
            build_directory=build_directory,
            verbose=True,
        )

    torch.distributed.barrier()


def build_blending_indices(
    dataset_index: numpy.ndarray,
    dataset_sample_index: numpy.ndarray,
    weights: list[float],
    num_datasets: int,
    size: int,
    verbose: bool,
) -> None:
    import helpers

    helpers.build_blending_indices(dataset_index, dataset_sample_index, weights, num_datasets, size, verbose)


def build_sample_idx(
    sizes: numpy.ndarray, doc_idx: numpy.ndarray, sequence_length: int, num_epochs: int, tokens_per_epoch: int
) -> numpy.ndarray:
    import helpers

    if doc_idx.dtype == numpy.int32:
        log_rank_0(logging.INFO, f"using int32 for sample idx")
        sample_idx = helpers.build_sample_idx_int32(sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch)
    elif doc_idx.dtype == numpy.int64:
        log_rank_0(logging.INFO, f"using int64 for sample idx")
        sample_idx = helpers.build_sample_idx_int64(sizes, doc_idx, sequence_length, num_epochs, tokens_per_epoch)
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
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w
