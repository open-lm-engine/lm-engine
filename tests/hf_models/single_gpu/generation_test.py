# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


import pytest
import torch

from ...test_common import skip_test_if_device_unavailable
from ..test_common import from_config, get_dense_test_config, get_dummy_inputs, get_moe_test_config


def test_generation_works(device: torch.device, position_embedding_type: str, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(device)

    if device.type == "cpu" and dtype == torch.float16:
        pytest.skip("LayerNormKernelImpl not implemented for Half")

    for config in [
        get_dense_test_config(position_embedding_type),
        get_moe_test_config(position_embedding_type),
    ]:
        model = from_config(config, dtype=dtype).to(device)
        model.eval()

        input_ids, attention_mask, _ = get_dummy_inputs(device)

        model.generate(input_ids=input_ids, attention_mask=attention_mask)
