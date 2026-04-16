# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import GPTBaseModel, GPTBasePreTrainedModel
from .weights import get_gpt_base_model_parallel_state_dict


class GPTBaseForCausalLM(GPTBasePreTrainedModel, CausalLMModelMixin):
    base_model_class = GPTBaseModel
    model_parallel_state_dict_function = get_gpt_base_model_parallel_state_dict
