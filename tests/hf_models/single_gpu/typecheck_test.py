# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels

from ..test_common import TestCommons


class TypeCheckTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")]))
    def test_no_attention_mask_flash_attention(self, device: torch.device) -> None:
        self.skip_test_if_device_unavailable(device)

        config = self.get_dense_test_config(
            attention_head_type="mha", position_embedding_type="learned_absolute", num_layers=8, num_attention_heads=32
        )
        model = self.from_config(config, use_padding_free_transformer=True).to(device)
        model.eval()

        input_ids, _, labels = self.get_dummy_inputs(device, return_list=True)
        attention_mask = [[1] * len(i) for i in input_ids]

        with enable_kernels([Kernel.flash_attention_2]):
            self.assertRaises(AssertionError, model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
