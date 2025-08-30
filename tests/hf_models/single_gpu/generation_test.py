# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


import torch
from parameterized import parameterized

from ..test_common import TestCommons


class GenerationTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(), TestCommons.get_position_embedding_types(), TestCommons.get_dtypes()
        )
    )
    def test_generation_works(self, device: torch.device, position_embedding_type: str, dtype: torch.dtype) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, dtype)

        for config in [
            self.get_dense_test_config(position_embedding_type),
            self.get_moe_test_config(position_embedding_type),
        ]:
            model = self.from_config(config, dtype=dtype).to(device)
            model.eval()

            input_ids, attention_mask, _ = self.get_dummy_inputs(device)

            model.generate(input_ids=input_ids, attention_mask=attention_mask)
