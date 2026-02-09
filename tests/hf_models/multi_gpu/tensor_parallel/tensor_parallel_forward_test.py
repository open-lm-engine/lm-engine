# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import torch
from parameterized import parameterized

from lm_engine.utils import is_flash_attention_2_available, is_flash_attention_3_available, torch_dtype_to_string

from ...test_common import TestCommons


class TensorParallelTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_position_embedding_types(),
            ["sdpa", "flash_attention_2", "flash_attention_3"],
            TestCommons.get_dtypes(),
            [False, True],
            [False, True],
        )
    )
    @TestCommons.slow_test
    def test_tensor_parallel_forward(
        self,
        position_embedding_type: str,
        attention_implementation: str,
        dtype: torch.dtype,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        if (attention_implementation, dtype) not in [
            ("sdpa", torch.float32),
            ("flash_attention_2", torch.float16),
        ]:
            self.skipTest("skipping test since running all takes too long")

        if attention_implementation == "flash_attention_2" and not is_flash_attention_2_available():
            self.skipTest("skipping test because flash attention 2 is unavailable")
        elif attention_implementation == "flash_attention_3" and not is_flash_attention_3_available():
            self.skipTest("skipping test because flash attention 3 is unavailable")

        if use_padding_free_transformer and attention_implementation != "flash_attention_2":
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

            if use_padding_free_transformer:
                command.append("--use-padding-free-transformer")

            if sequence_parallel:
                command.append("--sequence-parallel")

            subprocess.run(command, check=True)
