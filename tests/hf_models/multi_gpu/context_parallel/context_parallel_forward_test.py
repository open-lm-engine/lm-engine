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

from ....utils import skip_test_if_device_unavailable, slow_test


@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("attention_implementation", ["flash_attention_2", "flash_attention_3", "flash_attention_4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("load_balancing_method", [None])
@slow_test
def test_context_parallel_forward(
    position_embedding_type: str,
    attention_implementation: str,
    dtype: torch.dtype,
    load_balancing_method: str | None,
) -> None:
    skip_test_if_device_unavailable(torch.device("cuda"))

    for i, func in zip(
        range(2, 5),
        [is_flash_attention_2_available, is_flash_attention_3_available, is_flash_attention_4_available],
    ):
        if attention_implementation == f"flash_attention_{i}" and not func():
            pytest.skip(f"skipping test because flash attention {i} is unavailable")

    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node < 2:
        pytest.skip("context parallel requires at least 2 GPUs")

    with tempfile.TemporaryDirectory() as tmp_path:
        command = [
            "torchrun",
            "--nproc_per_node",
            str(gpus_per_node),
            "-m",
            "tests.hf_models.multi_gpu.context_parallel.context_parallel_forward",
            "--position-embedding-type",
            position_embedding_type,
            "--dtype",
            torch_dtype_to_string(dtype),
            "--attention-implementation",
            attention_implementation,
            "--tmp-path",
            tmp_path,
        ]

        if load_balancing_method is not None:
            command.extend(["--load-balancing-method", load_balancing_method])

        subprocess.run(command, check=True)
