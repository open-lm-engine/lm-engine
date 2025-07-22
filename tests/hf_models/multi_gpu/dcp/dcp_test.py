# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import torch
from parameterized import parameterized

from ...test_common import TestCommons


class DCPTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(["gqa", "mqa"], ["gelu", "geglu"], [(3, 2, 2), (3, 1, 4), (0, 4, 1)])
    )
    @TestCommons.slow_test
    def test_dcp(
        self,
        attention_head_type: str,
        activation_function: str,
        zero_stage_ddp_sizes: tuple[int, int, int],
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.dcp.dcp",
                "--train-config",
                "tests/hf_models/multi_gpu/dcp/train.yml",
                "--unshard-config",
                "tests/hf_models/multi_gpu/dcp/unshard.yml",
                "--attention-head-type",
                attention_head_type,
                "--activation-function",
                activation_function,
                "--tmp-path",
                tmp_path,
                "--zero-stage",
                str(zero_stage_ddp_sizes[0]),
                "--data-parallel-replication-world-size",
                str(zero_stage_ddp_sizes[1]),
                "--data-parallel-sharding-world-size",
                str(zero_stage_ddp_sizes[2]),
            ]

            subprocess.run(command, check=True)
