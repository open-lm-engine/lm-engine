# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import torch
from parameterized import parameterized

from lm_engine.hf_models import GPTBaseConfig, LadderResidualConfig
from lm_engine.utils import torch_dtype_to_string

from ...test_common import TestCommons


class TensorParallelTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            TestCommons.get_attention_implementations(),
            TestCommons.get_dtypes(),
            [False, True],
            [False, True],
            [GPTBaseConfig.model_type],
        )
        + TestCommons.make_args_matrix(
            ["gqa"],
            ["rope"],
            ["flash_attention_2"],
            [torch.float16],
            [False, True],
            [False, True],
            [LadderResidualConfig.model_type],
        )
    )
    @TestCommons.slow_test
    def test_tensor_parallel_forward(
        self,
        attention_head_type: str,
        position_embedding_type: str,
        attention_implementation: str,
        torch_dtype: torch.dtype,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        model_type: str,
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        if (attention_implementation, torch_dtype) not in [
            ("sdpa", torch.float32),
            ("flash_attention_2", torch.float16),
        ]:
            self.skipTest("skipping test since running all takes too long")

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
                "--attention-head-type",
                attention_head_type,
                "--position-embedding-type",
                position_embedding_type,
                "--torch-dtype",
                torch_dtype_to_string(torch_dtype),
                "--attention-implementation",
                attention_implementation,
                "--model-type",
                model_type,
                "--tmp-path",
                tmp_path,
            ]

            if use_padding_free_transformer:
                command.append("--use-padding-free-transformer")

            if sequence_parallel:
                command.append("--sequence-parallel")

            subprocess.run(command, check=True)
