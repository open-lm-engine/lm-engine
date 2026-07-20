# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass

import torch


@dataclass
class PositionInfo:
    position_ids: torch.Tensor | None
    rope_cos_sin: torch.Tensor | None
