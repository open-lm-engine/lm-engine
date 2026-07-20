# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os

import torch
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast as _HFCausalLMOutputWithPast

from .arguments import LoadArgs, MixedPrecisionArgs, UnshardingArgs
from .generation_cache import GenerationCache
from .loss import get_autoregressive_language_modeling_loss
from .mixins.dense.main import CausalLMModelMixin
from .model_config import CommonConfig
from .modeling_utils import AttentionMaskInfo, ParameterizedEmbedding, ParameterizedLinear, PositionInfo
from .parallel import ProcessGroupManager
from .parameter import (
    _INIT_MARKER,
    get_named_parameters_and_buffers,
    get_parameter_marker_maps,
    is_parameter_initialized,
    set_parameter_marker_maps,
)
from .utils import SafeTensorsWeightsManager, torch_dtype_to_string


_MODEL_TYPE_TO_CAUSAL_LM_CLASS: dict[str, type[CausalLMModelMixin]] = {}
_HF_USELESS_STUFF = [
    "revision",
    "subfolder",
    "gguf_file",
    "quantization_config",
    "max_memory",
    "name_or_path",
    "trust_remote_code",
    "adapter_kwargs",
]


class LLMAdapter_HFConfig(PretrainedConfig):
    """Wraps a lm_engine `CommonConfig` so it satisfies `transformers.PreTrainedModel`'s requirement that `config`
    be an instance of `PretrainedConfig`. Only ever used by `LLMAdapter_HF`, never for training."""

    model_type = "lm_engine_hf_adapter"

    @classmethod
    def from_common_config(cls, common_config: CommonConfig) -> LLMAdapter_HFConfig:
        kwargs = common_config.to_dict()
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("use_cache", True)

        return cls(**kwargs)


class LLMAdapter_HF(PreTrainedModel, GenerationMixin):
    """Adapts an already-built lm_engine `CausalLMModelMixin` model (e.g. `GPTBaseForCausalLM`) to the
    `transformers.PreTrainedModel` + `GenerationMixin` interface, so it can be dropped into `model.generate(...)`,
    `transformers.pipeline(...)`, or any HuggingFace-based evaluation harness without modification. It also owns
    checkpoint I/O (`from_pretrained`/`save_pretrained`) for all custom lm_engine architectures; the lm_engine
    training loop bypasses this class and constructs the raw model directly (see `register_hf.get_causal_lm_class`).

    This is an inference-only adapter: `forward()`/`generate()` assume a single-stage, non-padding-free model; use
    the lm_engine training loop directly for anything else (loss-scaled aux loss, `PipelineParallelOutput`, packed
    padding-free sequences).

    Example:
        hf_model = LLMAdapter_HF.from_pretrained(path, config=config)
        hf_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=20)
    """

    config_class = LLMAdapter_HFConfig
    base_model_prefix = "model"

    def __init__(self, model: CausalLMModelMixin) -> LLMAdapter_HF:
        # NOTE: pipeline-parallel and padding-free-transformer models are allowed through for now so callers
        # (e.g. `model_wrapper/base.py::get_loss`) can wrap one just to reach `get_output_embeddings()` etc.
        # `forward()`/`generate()` still assume a single-stage, non-padding-free model (no `PipelineParallelOutput`
        # handling, no `cu_seqlens`/`max_seqlen` params) — don't call them through this adapter on such a model.

        super().__init__(type(self).config_class.from_common_config(model.config))

        self.model = model

    @classmethod
    def _get_causal_lm_class(cls, config: CommonConfig) -> type[CausalLMModelMixin]:
        model_type = config.model_type
        assert model_type in _MODEL_TYPE_TO_CAUSAL_LM_CLASS, (
            f"unknown model_type ({model_type}) for LLMAdapter_HF, has `register_hf.register_model_classes` been "
            "called?"
        )
        return _MODEL_TYPE_TO_CAUSAL_LM_CLASS[model_type]

    @classmethod
    def _from_config(cls, config: CommonConfig, **kwargs) -> LLMAdapter_HF:
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        model = cls._get_causal_lm_class(config)(config, **kwargs)
        model = model.to(dtype)
        return cls(model)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: CommonConfig | None = None,
        dtype: torch.dtype = torch.float32,
        iteration: int | None = None,
        **kwargs,
    ) -> LLMAdapter_HF:
        assert len(model_args) == 0
        assert config is not None, "LLMAdapter_HF.from_pretrained needs `config` (an lm_engine CommonConfig)"

        model_class = cls._get_causal_lm_class(config)

        # drop useless stuff
        for k in _HF_USELESS_STUFF:
            kwargs.pop(k, None)

        num_pipeline_stages = kwargs.pop("num_pipeline_stages", 1)
        pipeline_stage_id = kwargs.pop("pipeline_stage_id", 0)

        if os.path.isfile(os.path.join(pretrained_model_name_or_path, "latest_checkpointed_iteration.json")):
            # lazy import avoids circular dependency (checkpointing → model_wrapper → hf_adapter)
            from .checkpointing import load_checkpoint_and_unshard

            assert not ProcessGroupManager.is_tensor_parallel_enabled()

            unshard_args = UnshardingArgs(
                load_args=LoadArgs(load_path=pretrained_model_name_or_path, iteration=iteration),
                mixed_precision_args=MixedPrecisionArgs(dtype=torch_dtype_to_string(dtype)),
                unsharded_path="",
            )

            model_wrapper, _, _ = load_checkpoint_and_unshard(unshard_args)
            model = model_wrapper.model
        else:
            assert iteration is None, "iteration should be None when loading from an unsharded checkpoint"

            if ProcessGroupManager.is_tensor_parallel_enabled() or num_pipeline_stages > 1:
                with torch.device("meta"):
                    model = model_class(
                        config, num_pipeline_stages=num_pipeline_stages, pipeline_stage_id=pipeline_stage_id, **kwargs
                    )

                for module in model.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

                marker_maps = get_parameter_marker_maps([model], extra_markers=[_INIT_MARKER])

                model = model.to(dtype=dtype)
                model = model.to_empty(device=torch.cuda.current_device())
                model.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(pretrained_model_name_or_path))
            else:
                model = model_class(config, **kwargs)
                marker_maps = get_parameter_marker_maps([model], extra_markers=[_INIT_MARKER])

                model = model.to(dtype=dtype)
                model.load_state_dict(SafeTensorsWeightsManager(pretrained_model_name_or_path).state_dict())

        device_map = kwargs.pop("device_map", {"": None})
        assert len(device_map) == 1
        assert len(kwargs) == 0

        model = model.to(device_map[""])
        set_parameter_marker_maps([model], marker_maps)

        for param_name, param in get_named_parameters_and_buffers(model):
            assert is_parameter_initialized(param), f"{param_name} is not initialized"

        return cls(model)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.model.config.save_pretrained(save_directory)
        SafeTensorsWeightsManager.save_state_dict(self.model.state_dict(), save_directory)

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # lm_engine models manage their own `GenerationCache`, created lazily on the first forward call when
        # `cache_params` is None. Returning False here stops `generate()` from pre-instantiating a `DynamicCache`
        # for us, the same escape hatch HF uses for architectures with a non-standard cache (e.g. Mamba).
        return False

    def can_generate(self) -> bool:
        return True

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: ParameterizedEmbedding) -> None:
        self.model.transformer.wte = value

    def get_output_embeddings(self) -> ParameterizedLinear:
        return self.model.transformer.wte if self.model._tied_word_embeddings else self.model.lm_head

    def set_output_embeddings(self, new_embeddings: ParameterizedLinear) -> None:
        if not self.model._tied_word_embeddings:
            self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,  # swallows extra kwargs GenerationMixin forwards (cache_position, output_attentions, ...)
    ) -> _HFCausalLMOutputWithPast:
        assert inputs_embeds is None, "LLMAdapter_HF does not support inputs_embeds"

        outputs = self.model(
            input_ids=input_ids,
            cache_params=past_key_values,
            attention_mask_info=AttentionMaskInfo(attention_mask=attention_mask),
            position_info=PositionInfo(position_ids=position_ids),
            use_cache=use_cache,
        )

        # loss is computed here, not by the wrapped model: the model's own loss path is tensor-parallel/aux-loss
        # aware and belongs to the training loop, whereas this adapter only ever runs plain, single-device inference
        loss = None
        if labels is not None:
            loss = get_autoregressive_language_modeling_loss(
                lm_logits=outputs.logits,
                labels=labels,
                reduction="mean",
                shift_logits_and_labels=True,
                tensor_parallel_enabled=False,
                use_padding_free_transformer=False,
            )

        return _HFCausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.cache_params,
        )


def build_hf_adapter_classes(config_class: type[CommonConfig]) -> type[LLMAdapter_HF]:
    """Builds an `LLMAdapter_HF` subclass for one lm_engine architecture, with `config_class` named to match
    `config_class` (e.g. `GPTBaseConfig`). `transformers.AutoModelForCausalLM.register` requires
    `model_class.config_class.__name__ == config_class.__name__`, and since a single `LLMAdapter_HF` wraps every
    lm_engine architecture, each one needs its own identically-named config shim to pass that check."""

    adapter_config_class = type(config_class.__name__, (LLMAdapter_HFConfig,), {})
    adapter_model_class = type(f"{config_class.__name__}LLMAdapter_HF", (LLMAdapter_HF,), {})
    adapter_model_class.config_class = adapter_config_class

    return adapter_model_class
