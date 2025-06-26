# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...mixins import PreTrainedModelMixin
from ..gpt_base import GPTBaseModel
from .config import DesyncResidualConfig
from .layer import DesyncResidualBlock


class DesyncResidualPreTrainedModel(PreTrainedModelMixin):
    config_class = DesyncResidualConfig
    layer_class = DesyncResidualBlock
    _no_split_modules = ["DesyncResidualBlock"]


class DesyncResidualModel(DesyncResidualPreTrainedModel, GPTBaseModel):
    def __init__(self, config: DesyncResidualConfig, **kwargs) -> None:
        self.tensor_parallel_size = config.pretraining_tensor_parallel_size

        super().__init__(config, **kwargs)

    def _get_rope_cos_sin(self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        rope_cos_sin = super()._get_rope_cos_sin(key_length, position_ids, dtype=dtype)

        if self.position_embedding_type == "rope":
            cos, sin = rope_cos_sin

            if position_ids.shape[0] != 1:
                cos = cos.repeat(self.tensor_parallel_size, 1, 1, 1)
                sin = sin.repeat(self.tensor_parallel_size, 1, 1, 1)

            return cos, sin
