# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin_TP
from .base import GPTBaseModel_TP, GPTBasePreTrainedModel_TP
from .weights import get_gpt_base_model_parallel_state_dict


class GPTBaseForCausalLM_TP(GPTBasePreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = GPTBaseModel_TP
    model_parallel_state_dict_function = get_gpt_base_model_parallel_state_dict
