# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import itertools

import torch
from parameterized import parameterized
from transformers import set_seed

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels

from ..test_common import TestCommons


SEED = 1234


class GPTBaseAttentionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")], TestCommons.get_position_embedding_types(), [torch.float16, torch.bfloat16]
        )
    )
    def test_sdpa_flash_attention_equivalence(
        self, device: torch.device, position_embedding_type: str, dtype: torch.dtype
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        config = self.get_dense_test_config(position_embedding_type, num_layers=1)

        model = self.from_config(config, dtype=dtype).to(device)
        model.eval()

        sdpa_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        sdpa_logits = sdpa_output.logits
        sdpa_loss = sdpa_output.loss

        with enable_kernels([Kernel.flash_attention_2]):
            flash_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            flash_logits = flash_output.logits
            flash_loss = flash_output.loss

        # we don't care about what happens on masked values (they don't match btw)
        sdpa_logits[attention_mask == 0] = 0
        flash_logits[attention_mask == 0] = 0

        self.assert_equal_tensors(
            sdpa_logits[attention_mask],
            flash_logits[attention_mask],
            False,
            rtol_float16=1e-3,
            atol_float16=3e-4,
            rtol_bfloat16=5e-3,
            atol_bfloat16=5e-3,
        )
        self.assert_equal_tensors(sdpa_loss, flash_loss, False, atol_float32=1.2e-4, rtol_float32=0)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")], TestCommons.get_position_embedding_types(), [torch.float16, torch.bfloat16]
        )
    )
    def test_flash_attention_equivalence_with_and_without_attention_masks(
        self, device: torch.device, position_embedding_type: str, dtype: torch.dtype
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        input_ids, _, labels = self.get_dummy_inputs(device)
        config = self.get_dense_test_config(position_embedding_type, num_layers=1)

        attention_mask = torch.ones_like(input_ids)

        model = self.from_config(config, dtype=dtype).to(device)
        model.eval()

        with enable_kernels([Kernel.flash_attention_2]):
            output_with_mask = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits_with_mask = output_with_mask.logits
            loss_with_mask = output_with_mask.loss

            output_without_mask = model(input_ids=input_ids, labels=labels)
            logits_without_mask = output_without_mask.logits
            loss_without_mask = output_without_mask.loss

        self.assert_equal_tensors(logits_with_mask, logits_without_mask, True)
        self.assert_equal_tensors(loss_with_mask, loss_without_mask, True)
