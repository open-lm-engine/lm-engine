# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...generation_cache import GenerationCache
from ...model_config import CommonConfig
from ...modeling_utils import (
    AttentionMaskInfo,
    Dropout,
    ParameterizedEmbedding,
    PositionInfo,
    RoPE,
    YaRNScaledRoPE,
    get_normalization_function,
)
from ...modeling_utils.init_utils import _get_std_for_embedding
from ...modeling_utils.io import BaseModelOutputWithPast
from ...parallel import ProcessGroupManager
from ...utils import divide_if_divisible, is_generation_cache_enabled
from .layer import Block


class PreTrainedModelMixin(nn.Module):
    config_class = None
    layer_class = Block
    _no_split_modules = ["Block"]

    def __init__(self, config: CommonConfig, *args, **kwargs) -> PreTrainedModelMixin:
        super().__init__()
        self.config = config

        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.num_pipeline_stages = kwargs.get("num_pipeline_stages", 1)
        self.pipeline_stage_id = kwargs.get("pipeline_stage_id", 0)

        self.is_first_stage = self.pipeline_stage_id == 0
        self.is_last_stage = self.pipeline_stage_id == self.num_pipeline_stages - 1
        self.is_pipeline_parallel_enabled = self.num_pipeline_stages > 1

        assert self.config_class is not None

        self.use_padding_free_transformer = kwargs.get("use_padding_free_transformer", False)
        self._tied_word_embeddings = config.tie_word_embeddings

        self.bos_token_id = self.config.bos_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = self.config.pad_token_id

        if self.is_pipeline_parallel_enabled and self._tied_word_embeddings:
            raise NotImplementedError()


class BaseModelMixin(PreTrainedModelMixin):
    def __init__(self, config: CommonConfig, **kwargs) -> BaseModelMixin:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_dim = config.rope_dim
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range

        self.layers_per_stage = divide_if_divisible(
            config.num_layers, self.num_pipeline_stages, "layers should be divisible by num_pipeline_stages"
        )

        self.layer_start_id = self.layers_per_stage * self.pipeline_stage_id
        self.layer_end_id = self.layers_per_stage * (self.pipeline_stage_id + 1)

        if self.is_first_stage:
            self.wte = ParameterizedEmbedding(
                config.vocab_size,
                self.embed_dim,
                std=_get_std_for_embedding(
                    initializer_range=self.initializer_range,
                    init_method=config.embedding_init_method,
                    embed_dim=self.embed_dim,
                ),
                use_padding_free_transformer=self.use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

            self.embedding_dropout = Dropout(
                config.embedding_dropout,
                use_padding_free_transformer=self.use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

        self.h = nn.ModuleDict(
            {
                str(i): self.layer_class(
                    config,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                    layer_idx=i,
                )
                for i in range(self.layer_start_id, self.layer_end_id)
            }
        )

        if self.is_last_stage:
            self.ln_f = get_normalization_function(
                config.normalization_function,
                self.embed_dim,
                eps=config.layer_norm_epsilon,
                use_padding_free_transformer=self.use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
            )

        self.position_embedding_type = config.position_embedding_type
        self.use_rope = self.position_embedding_type == "rope"
        self.use_learned_absolute = self.position_embedding_type == "learned_absolute"

        self._setup_positional_encoding()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
        use_cache: bool | None = None,
    ) -> BaseModelOutputWithPast:
        if self.is_first_stage:
            use_cache, hidden_states, attention_mask_info, position_info, cache_params = (
                self._prepare_a_bunch_of_stuff(
                    input_ids=input_ids,
                    cache_params=cache_params,
                    attention_mask_info=attention_mask_info,
                    position_info=position_info,
                    use_cache=use_cache,
                )
            )
        else:
            assert not ProcessGroupManager.is_context_parallel_enabled()
            assert cache_params is None
            assert attention_mask_info.attention_mask is None

            hidden_states = input_ids
            past_length = 0

            if self.use_padding_free_transformer:
                assert not ProcessGroupManager.is_context_parallel_enabled()
                key_length = attention_mask_info.max_seqlen
            else:
                key_length = (
                    hidden_states.size(1)
                    * (ProcessGroupManager.get_tensor_parallel_world_size() if self.self.sequence_parallel else 1)
                    * ProcessGroupManager.get_context_parallel_world_size()
                )

            position_info.reset_parameters(
                attention_mask=None,
                past_length=past_length,
                query_length=key_length - past_length,
                key_length=key_length,
                device=hidden_states.device,
            )

            if self.use_rope:
                position_info.rope_cos_sin = self._get_rope_cos_sin(key_length, position_info.position_ids)

        if is_generation_cache_enabled() and use_cache and cache_params is None:
            cache_params = GenerationCache()

        for layer_idx in range(self.layer_start_id, self.layer_end_id):
            block = self.h[str(layer_idx)]

            hidden_states = block(
                hidden_states,
                cache_params=cache_params,
                attention_mask_info=attention_mask_info,
                position_info=position_info,
            )

        if self.is_last_stage:
            hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache_params=cache_params)

    def _get_rope_cos_sin(self, key_length: int, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rope(key_length)
        cos = cos[position_ids]
        sin = sin[position_ids]
        return cos, sin

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
        use_cache: bool | None = None,
    ) -> tuple[bool, torch.Tensor, AttentionMaskInfo, PositionInfo, GenerationCache | None]:
        if use_cache is None:
            use_cache = False if self.use_padding_free_transformer else self.config.use_cache

        input_shape = input_ids.size()

        # special handling for padding free transformer with list inputs
        if self.use_padding_free_transformer:
            # for flash attention, there is no padding and we do packing
            # so, input_ids is of shape (s1 + s2 + ... + sb)
            batch_size = attention_mask_info.cu_seqlens.shape[0] - 1
        else:
            batch_size = input_shape[0]

        if self.use_padding_free_transformer:
            assert position_info.position_ids is not None, (
                "GPTBaseModel needs position_ids from outside when using flash attention with List[List[int]] "
                "inputs"
            )

        past_length = None
        query_length = None
        key_length = None
        if self.use_padding_free_transformer:
            max_seqlen = attention_mask_info.max_seqlen
            key_length = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
        else:
            past_length = 0 if cache_params is None else cache_params.get_seq_length()
            query_length = input_shape[-1] * (
                ProcessGroupManager.get_context_parallel_world_size()
                if ProcessGroupManager.is_context_parallel_enabled()
                else 1
            )
            key_length = past_length + query_length

        hidden_states = self.wte(input_ids)

        if self.use_rope or self.use_learned_absolute:
            position_info.reset_parameters(
                attention_mask=attention_mask_info.attention_mask,
                past_length=past_length,
                query_length=query_length,
                key_length=key_length,
                device=input_ids.device,
            )

        if self.use_learned_absolute:
            hidden_states = hidden_states + self.wpe(position_info.position_ids)

        hidden_states = self.embedding_dropout(hidden_states)

        if self.m_emb is not None:
            hidden_states = hidden_states * self.m_emb

        if self.use_rope:
            position_info.rope_cos_sin = self._get_rope_cos_sin(key_length, position_info.position_ids)

        attention_mask_info.reset_parameters(
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            dtype=hidden_states.dtype,
            device=input_ids.device,
        )

        return use_cache, hidden_states, attention_mask_info, position_info, cache_params

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == "learned_absolute":
            if self.is_first_stage:
                self.wpe = ParameterizedEmbedding(
                    max_position_embeddings,
                    self.embed_dim,
                    std=self.initializer_range,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )
        elif self.position_embedding_type == "rope":
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.rope_dim, max_position_embeddings=max_position_embeddings, base=self.config.rope_theta
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
