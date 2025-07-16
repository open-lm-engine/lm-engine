# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import DiffusionConfig


class DiffusionPreTrainedModel(PreTrainedModelMixin):
    config_class = DiffusionConfig


class DiffusionModel(DiffusionPreTrainedModel, BaseModelMixin): ...
