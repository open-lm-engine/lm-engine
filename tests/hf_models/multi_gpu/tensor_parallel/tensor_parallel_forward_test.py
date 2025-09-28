# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import torch
from parameterized import parameterized

from lm_engine.utils import torch_dtype_to_string

from ...test_common import TestCommons


class TensorParallelTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_position_embedding_types(),
            TestCommons.get_attention_implementations(),
            TestCommons.get_dtypes(),
            [False, True],
        )
    )
    @TestCommons.slow_test
    def test_tensor_parallel_forward(
        self,
        position_embedding_type: str,
        attention_implementation: str,
        dtype: torch.dtype,
        sequence_parallel: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        if (attention_implementation, dtype) not in [
            ("sdpa", torch.float32),
            ("flash_attention_2", torch.float16),
        ]:
            self.skipTest("skipping test since running all takes too long")

        if attention_implementation != "flash_attention_2":
            self.skipTest("skipping test since flash attention is needed for padding free transformer")

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.tensor_parallel.tensor_parallel_forward",
                "--position-embedding-type",
                position_embedding_type,
                "--dtype",
                torch_dtype_to_string(dtype),
                "--attention-implementation",
                attention_implementation,
                "--tmp-path",
                tmp_path,
            ]

            if sequence_parallel:
                command.append("--sequence-parallel")

            subprocess.run(command, check=True)
