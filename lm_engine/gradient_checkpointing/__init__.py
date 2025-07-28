# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch.nn as nn

from ..enums import GradientCheckpointingMethod
from ..utils import log_rank_0
from .block import block_checkpointing


_GRADIENT_CHECKPOINTING_METHODS = {GradientCheckpointingMethod.block: block_checkpointing}


def apply_gradient_checkpointing(
    model: nn.Module, gradient_checkpointing_method: GradientCheckpointingMethod, **kwargs
) -> None:
    log_rank_0(logging.INFO, "using activation checkpointing")

    checkpointing_function = _GRADIENT_CHECKPOINTING_METHODS[gradient_checkpointing_method]
    checkpointing_function(model, **kwargs)
