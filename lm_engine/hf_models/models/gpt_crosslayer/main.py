# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import GPTCrossLayerModel, GPTCrossLayerPreTrainedModel


class GPTCrossLayerForCausalLM(GPTCrossLayerPreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTCrossLayerModel
