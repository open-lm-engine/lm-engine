# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any, Callable

from ...utils import BaseArgs, divide_if_divisible
from .mlp import _MLPArgs, _MoEArgs
from .sequence_mixer import (
    _CausalConvolutionArgs,
    _GatedDeltaNetArgs,
    _GRUArgs,
    _M2RNNArgs,
    _Mamba2Args,
    _RNNArgs,
    _SoftmaxAttentionArgs,
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


_SEQUENCE_MIXER_CONFIG_CLASSES = {
    "causal_convolution": _CausalConvolutionArgs,
    "gru": _GRUArgs,
    "m2rnn": _M2RNNArgs,
    "mamba2": _Mamba2Args,
    "rnn": _RNNArgs,
    "softmax_attention": _SoftmaxAttentionArgs,
    "gated_deltanet": _GatedDeltaNetArgs,
}

_MLP_CONFIG_CLASSES = {"MLP": _MLPArgs, "MoE": _MoEArgs}
_ALL_INIT_METHODS = ["normal", "mup", "fan_in"]


class CommonConfig(BaseArgs):
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int = 768
    num_layers: int = 12
    embedding_dropout: float = 0
    normalization_function: str = "layernorm"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    position_embedding_type: str = "learned_absolute"
    rope_theta: int = 10000
    rope_scaling: dict | None = None
    m_emb: float | None = None
    m_width: float | None = None
    m_residual: float | None = None
    init_method: str = "normal"
    embedding_init_method: str = "normal"
    use_depth_scaled_init: bool = (True,)
    sequence_mixer_blocks: list[
        _SoftmaxAttentionArgs
        | _Mamba2Args
        | _GRUArgs
        | _RNNArgs
        | _M2RNNArgs
        | _CausalConvolutionArgs
        | _GatedDeltaNetArgs
    ]
    mlp_blocks: list[_MLPArgs | _MoEArgs]
    router_aux_loss_coef: float = 0.001
    tie_word_embeddings: bool = True
    rope_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        # check if enums are valid
        assert self.init_method in _ALL_INIT_METHODS
        assert self.embedding_init_method in _ALL_INIT_METHODS
        assert self.position_embedding_type in ["rope", "learned_absolute", "nope"]

        self._set_sequence_mixer_blocks()
        assert len(self.sequence_mixer_blocks) == self.num_layers

        self.rope_dim = rope_dim
        if self.rope_dim is None and position_embedding_type == "rope":
            assert (
                self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "sequence_mixer_type")
                == "softmax_attention"
            ), "specify rope_dim"

            self.rope_dim = self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "head_dim")

            if self.rope_dim is None:
                self.rope_dim = divide_if_divisible(
                    self.hidden_size,
                    self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
                    "",
                )

        self.mlp_blocks = mlp_blocks
        self._set_mlp_blocks()
        assert len(self.mlp_blocks) == self.num_layers

        self.router_aux_loss_coef = router_aux_loss_coef

    @_hold_base_args(key="sequence_mixer_blocks")
    @_hold_base_args(key="mlp_blocks")
    def to_json_string(self, use_diff: bool = True) -> str:
        return super().to_json_string(use_diff)

    def check_equal_for_all_and_get_value(
        self,
        key: str,
        key_block: str,
        expected_value: Any | None = None,
        sequence_mixer_type: str | None = None,
        mlp_type: str | None = None,
    ) -> Any:
        def _get(block, key):
            return block.get(key) if isinstance(block, dict) else getattr(block, key)

        blocks = getattr(self, key)
        if sequence_mixer_type is not None:
            blocks = filter(lambda block: _get(block, "sequence_mixer_type") == sequence_mixer_type, blocks)
            blocks = list(blocks)

        if mlp_type is not None:
            blocks = filter(lambda block: _get(block, "mlp_type") == mlp_type, blocks)
            blocks = list(blocks)

        value = _get(blocks[0], key_block)

        if expected_value is not None:
            assert value == expected_value, f"{value} {expected_value}"

        assert all([_get(block, key_block) == value for block in blocks])

        return value
