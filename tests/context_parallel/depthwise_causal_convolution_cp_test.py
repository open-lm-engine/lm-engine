# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import subprocess

import pytest
import torch

from lm_engine.utils import is_causal_conv1d_available

from ..utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("kernel_size", list(range(1, 5)))
@pytest.mark.parametrize("use_causal_conv1d", [False, True])
@slow_test
def test_depthwise_causal_convolution_cp(kernel_size: int, use_causal_conv1d: bool) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    if use_causal_conv1d and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    if use_causal_conv1d and kernel_size == 1:
        pytest.skip("causal_conv1d only supports kernel_size between 2 and 4")

    gpus_per_node = torch.cuda.device_count()
    if gpus_per_node < 2:
        pytest.skip("context parallel requires at least 2 GPUs")

    command = [
        "torchrun",
        "--nproc_per_node",
        str(gpus_per_node),
        "-m",
        "tests.hf_models.multi_gpu.context_parallel.depthwise_causal_convolution_cp",
        "--kernel-size",
        str(kernel_size),
    ]

    if use_causal_conv1d:
        command.append("--use-causal-conv1d")

    subprocess.run(command, check=True)
