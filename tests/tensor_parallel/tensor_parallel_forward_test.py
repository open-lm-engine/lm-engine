# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import subprocess
import tempfile

import pytest
import torch

from lm_engine.utils import (
    is_flash_attention_2_available,
    is_flash_attention_3_available,
    is_flash_attention_4_available,
    torch_dtype_to_string,
)

from ..utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_padding_free_transformer", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@slow_test
def test_tensor_parallel_forward(
    dtype: torch.dtype, use_padding_free_transformer: bool, sequence_parallel: bool
) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    if is_flash_attention_4_available():
        attention_implementation = "flash_attention_4"
    elif is_flash_attention_3_available():
        attention_implementation = "flash_attention_3"
    elif is_flash_attention_2_available():
        attention_implementation = "flash_attention_2"
    else:
        attention_implementation = "sdpa"

    if use_padding_free_transformer and attention_implementation not in [f"flash_attention_{i}" for i in range(2, 5)]:
        pytest.skip("skipping test since flash attention is needed for padding free transformer")

    gpus_per_node = torch.cuda.device_count()

    with tempfile.TemporaryDirectory() as tmp_path:
        command = [
            "torchrun",
            "--nproc_per_node",
            str(gpus_per_node),
            "-m",
            "tests.tensor_parallel.tensor_parallel_forward",
            "--position-embedding-type",
            "rope",
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
