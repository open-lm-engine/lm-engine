# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Multi-GPU tests spawn torchrun workers that use torch.cuda.set_device(local_rank).
# On machines with Exclusive Process compute mode this conflicts with the pytest
# process's own CUDA context (initialized by the first single-GPU test that runs).
# These tests must therefore run before any test that touches CUDA, which mirrors
# the ordering that existed on main (tests/hf_models/multi_gpu/ < single_gpu/).

_MULTI_GPU_DIRS = {"context_parallel", "dcp", "tensor_parallel", "unsharding"}


def pytest_collection_modifyitems(items: list) -> None:
    multi_gpu = [i for i in items if any(p in _MULTI_GPU_DIRS for p in i.nodeid.split("/"))]
    other = [i for i in items if i not in set(multi_gpu)]
    items[:] = multi_gpu + other
