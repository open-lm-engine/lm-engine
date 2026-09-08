# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import LadderResidualModel, LadderResidualPreTrainedModel


class LadderResidualForCausalLM(LadderResidualPreTrainedModel, CausalLMModelMixin):
    base_model_class = LadderResidualModel
