# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import subprocess

import pytest
import torch

from lm_engine.utils import is_mamba_2_ssm_available

from ..utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("use_mamba2_ssm", [True])
@slow_test
def test_mamba2_cp(use_mamba2_ssm: bool) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    if use_mamba2_ssm and not is_mamba_2_ssm_available():
        pytest.skip("mamba_ssm unavailable")

    gpus_per_node = torch.cuda.device_count()
    if gpus_per_node < 2:
        pytest.skip("context parallel requires at least 2 GPUs")

    command = [
        "torchrun",
        "--nproc_per_node",
        str(gpus_per_node),
        "-m",
        "tests.context_parallel.mamba2_cp",
    ]

    if use_mamba2_ssm:
        command.append("--use-mamba2-ssm")

    subprocess.run(command, check=True)
