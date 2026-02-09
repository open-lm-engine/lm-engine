# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin
from ..gpt_base import GPTBaseConfig


class GPTBasePreTrainedModel_TP(PreTrainedModelMixin):
    config_class = GPTBaseConfig


class GPTBaseModel_TP(GPTBasePreTrainedModel_TP, BaseModelMixin_TP): ...
