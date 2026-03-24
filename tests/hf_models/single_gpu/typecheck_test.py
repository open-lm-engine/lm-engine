# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels
from lm_engine.utils import is_flash_attention_2_available, is_flash_attention_3_available

from ...test_common import skip_test_if_device_unavailable
from ...utils import from_config, get_dense_test_config, get_dummy_inputs


@pytest.mark.parametrize("device", [torch.device("cuda")])
def test_no_attention_mask_flash_attention(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    config = get_dense_test_config(position_embedding_type="learned_absolute", num_layers=8, num_attention_heads=32)
    model = from_config(config, use_padding_free_transformer=True).to(device)
    model.eval()

    input_ids, _, labels = get_dummy_inputs(device, return_list=True)
    attention_mask = [[1] * len(i) for i in input_ids]

    kernel = None
    if is_flash_attention_3_available():
        kernel = Kernel.flash_attention_3
    if is_flash_attention_2_available():
        kernel = Kernel.flash_attention_2

    if kernel is None:
        pytest.skip("skipping test because flash attention 2 or 3 is unavailable")

    with enable_kernels([kernel]):
        pytest.assertRaises(AssertionError, model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
