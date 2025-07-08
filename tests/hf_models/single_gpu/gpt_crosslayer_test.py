# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized
from torch.testing import assert_close
from transformers import AutoConfig, AutoModelForCausalLM, set_seed

from lm_engine.enums import Kernel
from lm_engine.hf_models import convert_gpt_base_to_gpt_crosslayer
from lm_engine.kernels import enable_kernels

from ..test_common import TestCommons


SEED = 42


class GPTCrossLayerAttentionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
        )
    )
    def test_crosslayer_attention_equivalence(
        self,
        device: torch.device,
        attention_head_type: str,
        position_embedding_type: str,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        original_config = self.get_dense_test_config(
            attention_head_type, position_embedding_type=position_embedding_type
        )
        original_model = AutoModelForCausalLM.from_config(original_config).to(device)

        _, model = convert_gpt_base_to_gpt_crosslayer(original_config, original_model)
        model = model.to(device)

        original_model.eval()
        model.eval()

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)

        original_output = original_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )
        original_logits = original_output.logits
        original_loss = original_output.loss

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        logits = output.logits
        loss = output.loss

        assert logits.equal(original_logits)
        assert loss.equal(original_loss)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            TestCommons.get_dtypes(),
            [True, False],
        )
    )
    def test_generation_matches_gpt_base(
        self,
        device: torch.device,
        attention_head_type: str,
        position_embedding_type: str,
        torch_dtype: torch.dtype,
        use_cache: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        set_seed(SEED)

        original_config = self.get_dense_test_config(attention_head_type, position_embedding_type, num_layers=1)
        original_model = AutoModelForCausalLM.from_config(original_config, torch_dtype=torch_dtype).to(device)

        _, model = convert_gpt_base_to_gpt_crosslayer(original_config, original_model)
        model = model.to(device)

        original_model.eval()
        model.eval()

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        original_output = original_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert original_output.sequences.equal(output.sequences)

        for original_scores, scores in zip(original_output.scores, output.scores):
            assert_close(original_scores, scores, atol=2e-7, rtol=0)

    def from_config(self, config: AutoConfig, **kwargs) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

        kwargs.pop("torch_dtype")
        _, model = convert_gpt_base_to_gpt_crosslayer(config, model, **kwargs)

        use_padding_free_transformer = kwargs.pop("use_padding_free_transformer", False)
        if use_padding_free_transformer:
            assert model.use_padding_free_transformer

        assert len(kwargs) == 0

        return model
