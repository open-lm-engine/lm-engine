# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import PaLMConfig
from .layer import PaLMBlock


class PaLMPreTrainedModel(PreTrainedModelMixin):
    config_class = PaLMConfig
    layer_class = PaLMBlock
    _no_split_modules = ["PaLMBlock"]


class PaLMModel(PaLMPreTrainedModel, BaseModelMixin): ...
