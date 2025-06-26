# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from ..enums import DatasetSplit
from .base import BlendedDatasets


class BlendedDistributedSampler(DistributedSampler):
    """Data sampler used for training on multiple datasets according to the specified sampling proportions"""

    def __init__(
        self,
        dataset: BlendedDatasets,
        data_sampling_ratios: list[int],
        num_replicas: int,
        rank: int,
        ignore_sampling_proportion_for_validation: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.dataset: BlendedDatasets

        self.num_examples_in_each_dataset = self.dataset.get_num_examples_in_each_dataset()
        self.num_datasets = dataset.get_num_datasets()

        if self.dataset.split == DatasetSplit.val and ignore_sampling_proportion_for_validation:
            self.num_samples_by_dataset = self.num_examples_in_each_dataset
        else:
            self.num_samples_by_dataset = _get_num_samples_by_dataset(data_sampling_ratios, len(dataset))

        if shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        self.start_indices = np.cumsum([0] + self.num_examples_in_each_dataset[:-1]).tolist()
        # this is just needed to store the state
        self.index = 0

    def _get_indices_in_data_subset(self, num_samples_in_subset: int, subset_size: int) -> torch.Tensor:
        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)

        if num_samples_in_subset < subset_size:
            if self.shuffle:
                sampler = torch.randperm(subset_size, generator=self.generator)[:num_samples_in_subset]
            else:
                sampler = torch.arange(num_samples_in_subset)
        else:
            num_concats = num_samples_in_subset // subset_size
            padding = num_samples_in_subset - num_concats * subset_size
            sampler = list(range(subset_size)) * num_concats
            sampler = torch.tensor(sampler)

            if padding > 0:
                if self.shuffle:
                    padding_samples = torch.randperm(subset_size, generator=self.generator)[:padding]
                else:
                    padding_samples = torch.arange(padding)

                sampler = torch.cat([sampler, padding_samples])

        return sampler

    def __iter__(self) -> Iterator[int]:
        indices = []
        for dataset_index in range(self.num_datasets):
            sampler = self._get_indices_in_data_subset(
                self.num_samples_by_dataset[dataset_index], self.num_examples_in_each_dataset[dataset_index]
            )
            sampler += self.start_indices[dataset_index]

            indices.extend(sampler.tolist())

        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)

            # permute the sample indices
            indices = torch.tensor(indices)
            indices = indices[torch.randperm(len(indices), generator=self.generator)]
            indices = indices.tolist()

        if self.drop_last:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        else:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        self.index = 0
        for i in indices:
            self.index += 1
            yield i

        self.set_epoch(self.epoch + 1)

    def state_dict(self) -> dict:
        state_dict = {"epoch": self.epoch, "index": self.index}
        if self.shuffle:
            state_dict["generator"] = self.generator.get_state()

        return state_dict

    def load_state_dict(self, state_dict: dict) -> dict:
        self.set_epoch(state_dict["epoch"])
        if self.shuffle:
            self.generator.set_state(state_dict["generator"])

        for _ in self:
            if self.index == state_dict["index"]:
                break

    def __repr__(self) -> None:
        x = ""
        for i, dataset in enumerate(self.dataset.datasets):
            x += f"number of samples of {dataset.__class__.__name__} ({dataset.data_name}) in 1 epoch of the entire dataset = {self.num_samples_by_dataset[i]}\n"
            x += f"number of epochs of {dataset.__class__.__name__} ({dataset.data_name}) in 1 epoch of the entire dataset = {self.num_samples_by_dataset[i] / len(dataset)}\n\n"

        return x.rstrip()


def _get_num_samples_by_dataset(data_sampling_ratio: list[int], total_examples: int) -> list[int]:
    data_sampling_ratio = torch.tensor(data_sampling_ratio)
    num_samples_by_dataset = data_sampling_ratio / data_sampling_ratio.sum() * total_examples
    num_samples_by_dataset = num_samples_by_dataset.to(torch.long)
    num_samples_by_dataset[-1] = total_examples - num_samples_by_dataset[:-1].sum()
    return num_samples_by_dataset.tolist()
