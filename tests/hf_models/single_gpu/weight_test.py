# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from lm_engine.hf_models.modeling_utils import (
    get_sequence_mixer,
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)

from ..test_common import TestCommons


class WeightTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_attention_head_types()))
    def test_query_key_value_weight_loading_and_saving(self, attention_head_type: str) -> None:
        config = self.get_dense_test_config(attention_head_type, "learned_absolute")

        layer_idx = 1
        attention = get_sequence_mixer(config, True, False, layer_idx)
        num_key_value_heads = config.sequence_mixer_blocks[layer_idx].num_key_value_heads

        state_dict = attention.state_dict()
        num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")

        c_attn_weight = state_dict["c_attn.weight"]
        query_key_value_weight = self._split_and_interleave(
            c_attn_weight,
            num_attention_heads,
            num_key_value_heads,
            config.hidden_size // num_attention_heads,
            attention_head_type,
        )
        assert (c_attn_weight == query_key_value_weight).all()

        c_attn_bias = state_dict["c_attn.bias"]
        query_key_value_bias = self._split_and_interleave(
            c_attn_bias,
            num_attention_heads,
            num_key_value_heads,
            config.hidden_size // num_attention_heads,
            attention_head_type,
        )
        assert (c_attn_bias == query_key_value_bias).all()

    def _split_and_interleave(
        self,
        c_attn_tensor: torch.Tensor,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_head_type: str,
    ) -> torch.Tensor:
        query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
            c_attn_tensor,
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )

        query_key_value_weight = interleave_query_key_value_tensor_for_attention(
            query_weight,
            key_weight,
            value_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )

        return query_key_value_weight
