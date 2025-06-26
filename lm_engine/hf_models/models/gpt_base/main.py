# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import GPTBaseModel, GPTBasePreTrainedModel


class GPTBaseForCausalLM(GPTBasePreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTBaseModel
