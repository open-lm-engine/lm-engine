# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .config import CommonConfig
from .loss import get_autoregressive_language_modeling_loss, is_aux_loss_zero
from .mixins import CausalLMOutputWithPast, PipelineParallelInput, PipelineParallelOutput
from .model_conversion import export_to_huggingface, import_from_huggingface
from .models import (
    GPTBaseConfig,
    GPTBaseForCausalLM,
    GPTBaseForCausalLM_TP,
    GPTBaseModel,
    GPTBaseModel_TP,
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    LadderResidualConfig,
    LadderResidualForCausalLM,
    LadderResidualModel,
    PaLMConfig,
    PaLMForCausalLM,
    PaLMModel,
)
from .parameter import (
    _INIT_MARKER,
    get_parameter_marker_maps,
    is_parameter_initialized,
    is_parameter_with_mup_learning_rate,
    is_parameter_with_no_weight_decay,
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
    set_parameter_marker_maps,
)
from .register_hf import get_model_parallel_class, is_custom_model, register_model_classes
from .unshard import fix_unsharded_state_dict, unshard_tensor_parallel_state_dicts
from .utils import convert_padding_free_lists_to_tensors, disable_generation_cache


register_model_classes()
