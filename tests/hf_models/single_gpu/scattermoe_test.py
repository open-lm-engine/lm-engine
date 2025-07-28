# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized
from transformers import set_seed

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels

from ..test_common import TestCommons


SEED = 1234


class ScatterMoETest(TestCommons):
    @parameterized.expand(TestCommons.get_dtypes())
    def test_scattermoe(self, torch_dtype: torch.dtype) -> None:
        device = torch.device("cuda")
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        config = self.get_moe_test_config("mha", "rope", num_layers=1, add_bias=False)

        model = self.from_config(config, torch_dtype=torch_dtype).to(device)
        model.eval()

        naive_output = model(input_ids=input_ids, attention_mask=attention_mask)

        with enable_kernels([Kernel.scattermoe]):
            scatter_output = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assert_equal_tensors(
            naive_output.logits,
            scatter_output.logits,
            False,
            rtol_float32=1e-3,
            atol_float32=2e-4,
            rtol_float16=0,
            atol_float16=2.5e-4,
            rtol_bfloat16=0,
            atol_bfloat16=2e-3,
        )
