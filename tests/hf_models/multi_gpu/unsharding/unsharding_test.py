# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import torch
from parameterized import parameterized

from ...test_common import TestCommons


class UnshardingTest(TestCommons):
    @parameterized.expand(["gelu", "geglu"])
    @TestCommons.slow_test
    def test_unsharding(self, activation_function: str) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "uv run torchrun",
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
