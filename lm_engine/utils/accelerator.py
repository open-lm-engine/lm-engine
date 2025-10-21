from __future__ import annotations

import torch
from enums import Enum


class Accelerator(Enum):
    cuda = "cuda"
    tpu = "tpu"

    @classmethod
    def get_accelerator_from_tensor(x: torch.Tensor) -> Accelerator:
        device = x.device.type
        return Accelerator.cuda if device == "cuda" else Accelerator.tpu
