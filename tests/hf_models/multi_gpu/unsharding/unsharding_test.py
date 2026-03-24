# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import pytest
import torch
from parameterized import parameterized

from ....utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("activation_function", ["gelu", "geglu"])
@slow_test
def test_unsharding(activation_function: str) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    gpus_per_node = torch.cuda.device_count()

    with tempfile.TemporaryDirectory() as tmp_path:
        command = [
            "torchrun",
            "--nproc_per_node",
            str(gpus_per_node),
            "-m",
            "tests.hf_models.multi_gpu.unsharding.unsharding",
            "--activation-function",
            activation_function,
            "--tmp-path",
            tmp_path,
        ]

        subprocess.run(command, check=True)
