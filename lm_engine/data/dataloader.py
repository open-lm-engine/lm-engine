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


class TrainiumDataLoader(ResumableDataLoader):
    def __init__(self, dataset: Dataset, batch_sampler: BatchSampler) -> TrainiumDataLoader:
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield {
                "text": torch.tensor(
                    np.array([self.dataset[i]["text"] for i in batch_indices]), device=Accelerator.get_current_device()
                )
            }
