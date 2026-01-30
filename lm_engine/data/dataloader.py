# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from ..utils import Accelerator


class ResumableDataLoader(DataLoader):
    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict.get("dataset"))
        self.sampler.load_state_dict(state_dict.get("sampler"))


class DummyDataLoader:
    def __iter__(self):
        B = 1
        S = 4096
        yield {"text": torch.tensor(B * [list(range(S + 1))], device=Accelerator.get_current_device())}
