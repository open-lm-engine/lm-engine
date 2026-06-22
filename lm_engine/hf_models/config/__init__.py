# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import json
import os
from typing import Any

from ...arguments import BaseArgs
from ...utils import divide_if_divisible
from .mlp import _MLPArgs, _MoEArgs
from .sequence_mixer import _GatedDeltaNetArgs, _GRUArgs, _M2RNNArgs, _Mamba2Args, _RNNArgs, _SoftmaxAttentionArgs


_ALL_INIT_METHODS = ["normal", "mup", "fan_in"]

# Keys added by HuggingFace internals that are not part of our config schema
_HF_META_KEYS = frozenset(
    {
        "architectures",
        "_from_auto",
        "_commit_hash",
        "transformers_version",
        "auto_map",
    }
)


class CommonConfig(BaseArgs):
    model_type: str = None
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_layers: int
    embedding_dropout: float = 0
    normalization_function: str
    layer_norm_epsilon: float = 1e-5
    initializer_range: float
    use_cache: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    position_embedding_type: str
    rope_theta: int
    rope_scaling: dict | None
    m_emb: float | None
    m_width: float | None
    m_residual: float | None
    init_method: str
    embedding_init_method: str
    use_depth_scaled_init: bool
    sequence_mixer_blocks: list[
        _SoftmaxAttentionArgs | _Mamba2Args | _GRUArgs | _RNNArgs | _M2RNNArgs | _GatedDeltaNetArgs
    ]
    mlp_blocks: list[_MLPArgs | _MoEArgs]
    router_aux_loss_coef: float
    tie_word_embeddings: bool
    rope_dim: int | None
    # HuggingFace compatibility fields
    dtype: str | None = None
    name_or_path: str = ""
    is_encoder_decoder: bool = False

    def model_post_init(self, __context: Any) -> None:
        # check if enums are valid
        assert self.init_method in _ALL_INIT_METHODS
        assert self.embedding_init_method in _ALL_INIT_METHODS
        assert self.position_embedding_type in ["rope", "learned_absolute", "nope"]

        assert len(self.sequence_mixer_blocks) == self.num_layers
        assert len(self.mlp_blocks) == self.num_layers

        if self.rope_dim is None and self.position_embedding_type == "rope":
            assert (
                self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "sequence_mixer_type")
                == "softmax_attention"
            ), "specify rope_dim"

            self.rope_dim = self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "head_dim")

            if self.rope_dim is None:
                self.rope_dim = divide_if_divisible(
                    self.hidden_size,
                    self.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
                )

    def to_dict(self) -> dict:
        # Only serialize declared Pydantic fields, not extra attrs set by HuggingFace internals
        from copy import deepcopy
        from enum import Enum

        result = {}
        for key in type(self).model_fields:
            value = getattr(self, key)
            if isinstance(value, BaseArgs):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                serialized = []
                for v in value:
                    if isinstance(v, BaseArgs):
                        serialized.append(v.to_dict())
                    elif isinstance(v, Enum):
                        serialized.append(v.value)
                    else:
                        serialized.append(deepcopy(v))
                result[key] = serialized
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = deepcopy(value)
        return result

    def get_text_config(self, decoder=None, encoder=None) -> "CommonConfig":
        return self

    def _get_non_default_generation_parameters(self) -> dict:
        return {}

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        # Add HuggingFace-expected metadata
        config_dict["architectures"] = [type(self).__name__.replace("Config", "ForCausalLM")]
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.to_json_string())

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> CommonConfig:
        kwargs.pop("return_unused_kwargs", None)
        config_dict = {k: v for k, v in config_dict.items() if k not in _HF_META_KEYS}
        # name_or_path is set by HF but not part of our Pydantic schema — store post-init
        name_or_path = config_dict.pop("name_or_path", "")
        config = cls(**config_dict)
        config.name_or_path = name_or_path
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> CommonConfig:
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, **kwargs)

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
