# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from dataclasses import dataclass

import torch
from transformers.modeling_outputs import ModelOutput

from ..cache import GenerationCache


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    cache_params: GenerationCache | None = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None
    logits: torch.Tensor | None = None
    cache_params: GenerationCache | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class PipelineParallelInput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None


@dataclass
class PipelineParallelOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None
