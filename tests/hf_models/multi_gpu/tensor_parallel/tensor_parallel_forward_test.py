# **************************************************
# Copyright (c) 2025, Mayank Mishra
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

from ....utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("attention_implementation", ["sdpa", "flash_attention_2", "flash_attention_3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_padding_free_transformer", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@slow_test
def test_tensor_parallel_forward(
    position_embedding_type: str,
    attention_implementation: str,
    dtype: torch.dtype,
    use_padding_free_transformer: bool,
    sequence_parallel: bool,
) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    if (attention_implementation, dtype) not in [
        ("sdpa", torch.float32),
        ("flash_attention_2", torch.float16),
        ("flash_attention_3", torch.float16),
        ("flash_attention_4", torch.float16),
    ]:
        pytest.skip("skipping test since running all takes too long")

    for i, func in zip(
        range(2, 5), [is_flash_attention_2_available, is_flash_attention_3_available, is_flash_attention_4_available]
    ):
        if attention_implementation == f"flash_attention_{i}" and not func():
            pytest.skip(f"skipping test because flash attention {i} is unavailable")

    if use_padding_free_transformer and attention_implementation not in [f"flash_attention_{i}" for i in range(2, 5)]:
        pytest.skip("skipping test since flash attention is needed for padding free transformer")

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
