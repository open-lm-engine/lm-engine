# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from lm_engine.hf_models.config import _Mamba2Args

from ..test_common import TestCommons


class ModelConversionTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices(), [True, False], [True, False]))
    def test_llama_model_conversion(self, device: torch.device, add_bias: bool, use_interleaved_weights: bool) -> None:
        lm_engine_config = self.get_dense_test_config(
            "rope", add_bias=add_bias, activation_function="swiglu", normalization_function="rmsnorm"
        )

        for block in lm_engine_config.mlp_blocks:
            block.use_interleaved_weights = use_interleaved_weights

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="llama",
            device=device,
            exact_match=False,
            use_interleaved_weights=use_interleaved_weights,
        )

    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices(), [True, False], [True, False]))
    def test_granite_model_conversion(
        self, device: torch.device, add_bias: bool, use_interleaved_weights: bool
    ) -> None:
        lm_engine_config = self.get_dense_test_config(
            "rope",
            add_bias=add_bias,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        for block in lm_engine_config.mlp_blocks:
            block.use_interleaved_weights = use_interleaved_weights

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="granite",
            device=device,
            exact_match=False,
            use_interleaved_weights=use_interleaved_weights,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), [True, False], [True, False], [True, False])
    )
    def test_granitemoehybrid_model_conversion(
        self,
        device: torch.device,
        is_moe: bool,
        use_interleaved_weights: bool,
        use_interleaved_weights_for_shared_experts: bool,
    ) -> None:
        if is_moe:
            lm_engine_config = self.get_moe_test_config(
                "nope",
                add_bias=False,
                shared_n_inner=64,
                activation_function="swiglu",
                normalization_function="rmsnorm",
                m_emb=2,
                m_width=2,
            )
        else:
            lm_engine_config = self.get_dense_test_config(
                "nope",
                add_bias=False,
                activation_function="swiglu",
                normalization_function="rmsnorm",
                m_emb=2,
                m_width=2,
            )

        for layer in range(lm_engine_config.num_layers):
            if layer % 2 == 0:
                lm_engine_config.sequence_mixer_blocks[layer] = _Mamba2Args(intermediate_size=256)

        for block in lm_engine_config.mlp_blocks:
            block.use_interleaved_weights = use_interleaved_weights
            block.use_interleaved_weights_for_shared_experts = use_interleaved_weights_for_shared_experts

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="granitemoehybrid",
            device=device,
            exact_match=False,
            compare_loss=False,
            logits_atol_float32=2.5e-5,
            use_interleaved_weights=use_interleaved_weights,
            use_interleaved_weights_for_shared_experts=use_interleaved_weights_for_shared_experts,
        )
