# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass

import torch


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor | None = None
    cache_params: tuple[tuple[torch.Tensor]] | None = None


@dataclass
class CausalLMOutputWithPast:
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None
    logits: torch.Tensor | None = None
    cache_params: tuple[tuple[torch.Tensor]] | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class PipelineParallelInput:
    hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None


@dataclass
class PipelineParallelOutput:
    hidden_states: torch.Tensor | None = None
    aux_loss: torch.Tensor | float | None = None
