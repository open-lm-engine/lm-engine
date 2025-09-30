# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel

from ...cache import GenerationCache
from ...config import CommonConfig
from ...mask import AttentionMaskInfo
from ...modeling_utils import ParameterizedEmbedding, RoPE, YaRNScaledRoPE, get_normalization_function
from ...utils import is_generation_cache_enabled
from ..modeling_outputs import BaseModelOutputWithPast
from .layer import Block


class PreTrainedModelMixin(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = None
    layer_class = Block
    base_model_prefix = "transformer"
    causal = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config: CommonConfig, *args, **kwargs) -> PreTrainedModelMixin:
        super().__init__(config, *args, **kwargs)

        assert self.config_class is not None
        self.generation_config = GenerationConfig.from_model_config(self.config)

        self._tied_word_embeddings = config.tie_word_embeddings
        self._has_mamba2 = any([block.sequence_mixer_type == "mamba2" for block in self.config.sequence_mixer_blocks])

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


class BaseModelMixin(PreTrainedModelMixin):
    mask_value = None

    def __init__(self, config: CommonConfig, **kwargs) -> BaseModelMixin:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.sequence_mixer_block_types = [
            config.sequence_mixer_blocks[i].sequence_mixer_type for i in range(config.num_layers)
        ]

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.embedding_dropout = (
            nn.Identity() if config.embedding_dropout == 0 else nn.Dropout(config.embedding_dropout)
        )
        self.h = nn.ModuleList([self.layer_class(config, layer_idx=i) for i in range(config.num_layers)])
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask_info: AttentionMaskInfo,
        past_key_values: GenerationCache | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> BaseModelOutputWithPast:
        use_cache, hidden_states, position_ids, rope_cos_sin = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            max_seqlen=attention_mask_info.get_max_seqlen(),
            position_ids=position_ids,
            use_cache=use_cache,
        )

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        for block in self.h:
            hidden_states: torch.Tensor = block(
                hidden_states,
                attention_mask_info=attention_mask_info,
                past_key_values=past_key_values,
                rope_cos_sin=rope_cos_sin,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _get_position_ids(
        self, attention_mask: torch.Tensor, past_length: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position_embedding_type == "rope":
            cos, sin = self.rope(key_length, dtype=dtype)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            return cos, sin

    def _get_initial_hidden_state(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        hidden_state = self.wte(input_ids)

        if self.position_embedding_type == "learned_absolute":
            hidden_state = hidden_state + self.wpe(position_ids)

        hidden_state = self.embedding_dropout(hidden_state)

        if self.m_emb is not None:
            hidden_state = hidden_state * self.m_emb

        return hidden_state

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert position_ids is not None, (
            "GPTBaseModel needs position_ids from outside when using flash attention with List[List[int]] " "inputs"
        )

        key_length = max_seqlen

        hidden_states = self._get_initial_hidden_state(input_ids, position_ids)
        rope_cos_sin = self._get_rope_cos_sin(key_length, position_ids, dtype=hidden_states.dtype)

        return use_cache, hidden_states, position_ids, rope_cos_sin

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == "learned_absolute":
            self.wpe = ParameterizedEmbedding(max_position_embeddings, self.embed_dim, std=self.initializer_range)
        elif self.position_embedding_type == "rope":
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.rope_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.rope_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        elif self.position_embedding_type == "nope":
            pass
        else:
            raise NotImplementedError()

    def _get_mamba_mask(
        self, attention_mask: torch.Tensor | None, past_key_values: GenerationCache
    ) -> torch.Tensor | None:
        mamba_mask = attention_mask
        if (
            past_key_values is None
            or past_key_values.get_seq_length() > 0
            or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            mamba_mask = None

        return mamba_mask
