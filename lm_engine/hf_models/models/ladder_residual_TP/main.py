# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin_TP
from ..gpt_base_TP.weights import get_gpt_base_model_parallel_state_dict
from .base import LadderResidualModel_TP, LadderResidualPreTrainedModel_TP


class LadderResidualForCausalLM_TP(LadderResidualPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = LadderResidualModel_TP
    model_parallel_state_dict_function = get_gpt_base_model_parallel_state_dict
