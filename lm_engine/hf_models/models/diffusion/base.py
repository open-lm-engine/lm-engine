# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import DiffusionConfig


class DiffusionPreTrainedModel(PreTrainedModelMixin):
    config_class = DiffusionConfig


class DiffusionModel(DiffusionPreTrainedModel, BaseModelMixin):
    def __init__(self, config, **kwargs):
        if "mask_token_id" in kwargs:
            self.mask_token_id = kwargs.pop("mask_token_id")
        super().__init__(config, **kwargs)

    def _get_initial_hidden_state(self, input_ids, position_ids):
        hidden_state = super()._get_initial_hidden_state(input_ids, position_ids)
        # mask = (input_ids == self.mask_token_id)[:, None]
        # hidden_state = hidden_state.masked_fill_(mask, 0)
        return hidden_state
