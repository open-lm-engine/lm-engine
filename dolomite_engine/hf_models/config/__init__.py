# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from copy import deepcopy
from typing import Any, Callable

from transformers import PretrainedConfig

from ...utils import BaseArgs, divide_if_divisible
from .mlp import _MLPArgs, _MoEArgs
from .sequence_mixer import (
    _CausalConvolution,
    _GRUArgs,
    _Mamba2Args,
    _MultiHeadLatentAttentionArgs,
    _RNNArgs,
    _SoftmaxAttentionArgs,
    _StickbreakingAttentionArgs,
)


def _hold_base_args(key: str) -> Callable:
    def _holded_function(function: Callable) -> Callable:
        def _run(self, *args, **kwargs):
            value: list[BaseArgs] = getattr(self, key)
            setattr(self, key, [i.to_dict() if isinstance(i, BaseArgs) else i for i in value])
            output = function(self, *args, **kwargs)
            setattr(self, key, value)
            return output

        return _run

    return _holded_function


def _update_with_key_value(block: dict, kwargs: dict, key: str) -> None:
    if key in block:
        kwargs[key] = block.pop(key)


_NAKED_DISALLOWED_ARGS = [
    "activation_function",
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
    "intermediate_size",
    "shared_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "add_bias",
    "attention_blocks",
    "num_key_value_heads",
    "attention_head_type",
    "attention_multiplier",
    "n_embd",
    "n_head",
    "n_inner",
    "n_layer",
    "n_positions",
    "scale_attn_weights",
    "num_attention_heads",
]

_SEQUENCE_MIXER_CONFIG_CLASSES = {
    "causal_convolution": _CausalConvolution,
    "gru": _GRUArgs,
    "mamba2": _Mamba2Args,
    "multihead_latent_attention": _MultiHeadLatentAttentionArgs,
    "rnn": _RNNArgs,
    "stickbreaking_attention": _StickbreakingAttentionArgs,
    "softmax_attention": _SoftmaxAttentionArgs,
}

_MLP_CONFIG_CLASSES = {"MLP": _MLPArgs, "MoE": _MoEArgs}


class CommonConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50304,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_layers: int = 12,
        embedding_dropout: float = 0,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict | None = None,
        m_emb: float | None = None,
        m_width: float | None = None,
        m_residual: float | None = None,
        init_method: str = "normal",
        sequence_mixer_blocks: list[dict] = None,
        mlp_blocks: list[dict] = None,
        router_aux_loss_coef: float = 0.001,
        tie_word_embeddings: bool = True,
        rope_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.normalization_function = normalization_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.m_emb = m_emb
        self.m_width = m_width
        self.m_residual = m_residual
        self.init_method = init_method

        # check if enums are valid
        assert init_method in ["normal", "mup"]
        assert position_embedding_type in ["rope", "learned_absolute", "nope"]

        self.sequence_mixer_blocks = sequence_mixer_blocks
        self._set_sequence_mixer_blocks()
        assert len(self.sequence_mixer_blocks) == self.num_layers

        self.rope_dim = rope_dim
        if self.rope_dim is None and position_embedding_type == "rope":
            assert (
                self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "sequence_mixer_type")
                == "softmax_attention"
            ), "specify rope_dim"

            self.rope_dim = divide_if_divisible(
                self.hidden_size,
                self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
                "",
            )

        self.mlp_blocks = mlp_blocks
        self._set_mlp_blocks()
        assert len(self.mlp_blocks) == self.num_layers

        self.router_aux_loss_coef = router_aux_loss_coef

        for i in _NAKED_DISALLOWED_ARGS:
            assert i not in kwargs, f"found naked argument ({i})"

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @_hold_base_args(key="sequence_mixer_blocks")
    @_hold_base_args(key="mlp_blocks")
    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs) -> None:
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    @_hold_base_args(key="sequence_mixer_blocks")
    @_hold_base_args(key="mlp_blocks")
    def to_json_string(self, use_diff: bool = True) -> str:
        return super().to_json_string(use_diff)

    def check_equal_for_all_and_get_value(
        self, key: str, key_block: str, expected_value: Any | None = None, sequence_mixer_type: str | None = None
    ) -> Any:
        def _get(block, key):
            return block.get(key) if isinstance(block, dict) else getattr(block, key)

        blocks = getattr(self, key)
        if sequence_mixer_type is not None:
            blocks = filter(lambda block: _get(block, "sequence_mixer_type") == sequence_mixer_type, blocks)
            blocks = list(blocks)

        value = _get(blocks[0], key_block)

        if expected_value is not None:
            assert value == expected_value, f"{value} {expected_value}"

        assert all([_get(block, key_block) == value for block in blocks])

        return value

    def _set_sequence_mixer_blocks(self) -> None:
        if self.sequence_mixer_blocks is None:
            self.sequence_mixer_blocks = [{} for _ in range(self.num_layers)]

        sequence_mixer_blocks: list[
            _CausalConvolution
            | _GRUArgs
            | _Mamba2Args
            | _MultiHeadLatentAttentionArgs
            | _RNNArgs
            | _SoftmaxAttentionArgs
            | _StickbreakingAttentionArgs
        ] = []
        for i in range(self.num_layers):
            sequence_mixer_block = deepcopy(self.sequence_mixer_blocks[i])
            sequence_mixer_type = sequence_mixer_block.pop("sequence_mixer_type", "softmax_attention")

            if sequence_mixer_type == "mamba2":
                sequence_mixer_block["intermediate_size"] = sequence_mixer_block.pop(
                    "intermediate_size", 2 * self.hidden_size
                )

            sequence_mixer_blocks.append(_SEQUENCE_MIXER_CONFIG_CLASSES[sequence_mixer_type](**sequence_mixer_block))

        self.sequence_mixer_blocks = sequence_mixer_blocks

    def _set_mlp_blocks(self) -> None:
        if self.mlp_blocks is None:
            self.mlp_blocks = [{} for _ in range(self.num_layers)]

        mlp_blocks: list[_MLPArgs | _MoEArgs] = []
        for i in range(self.num_layers):
            mlp_block = deepcopy(self.mlp_blocks[i])
            mlp_block["intermediate_size"] = mlp_block.pop("intermediate_size", 4 * self.hidden_size)

            mlp_type = mlp_block.pop("mlp_type", "MLP")
            mlp_blocks.append(_MLP_CONFIG_CLASSES[mlp_type](**mlp_block))

        self.mlp_blocks = mlp_blocks
