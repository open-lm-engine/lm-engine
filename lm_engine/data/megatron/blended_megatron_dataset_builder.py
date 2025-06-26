# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import math
from copy import deepcopy
from typing import Any, List, Optional, Type, Union

import numpy
import torch
import torch.distributed
from transformers import AutoTokenizer

from ...utils import ProcessGroupManager
from .blended_dataset import BlendedDataset
from .blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from .indexed_dataset import MMapIndexedDataset
from .megatron_dataset import MegatronDataset
from .utils import Split, normalize


DistributedDataset = Union[BlendedDataset, MegatronDataset, MMapIndexedDataset]


class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset
        sizes (List[int]): The minimum number of total samples to draw from each split, varies
        with blend
        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: Type[MegatronDataset],
        sizes: List[int],
        config: BlendedMegatronDatasetConfig,
        tokenizer: AutoTokenizer,
    ):
        self.cls = cls
        self.sizes = sizes
        self.config = config
        self.tokenizer = tokenizer

    def build(self) -> List[Optional[Union[BlendedDataset, MegatronDataset]]]:
        """Build all dataset splits according to the provided blend(s)

        This method is distributed-aware and must be called on all ranks.

        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions.

        Returns:
            List[Optional[Union[BlendedDataset, MegatronDataset]]]: A list of either
            MegatronDataset or BlendedDataset (or None) per split
        """

        if getattr(self.config, "blend"):
            blend = getattr(self.config, "blend")
            split = getattr(self.config, "split_vector")

            # Blend consists of a single prefix
            if len(blend) == 1:
                return self._build_megatron_dataset_splits(blend[0], split, self.sizes)

            # Blend consists of multiple weights and prefixes
            (
                prefix_per_dataset,
                weight_per_dataset,
                sizes_per_dataset,
            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)

            megatron_datasets = [[] for _ in range(len(Split))]

            for i in range(len(prefix_per_dataset)):
                megatron_datasets_split = self._build_megatron_dataset_splits(
                    prefix_per_dataset[i], split, sizes_per_dataset[i]
                )
                for j in range(len(megatron_datasets_split)):
                    megatron_datasets[j].append(megatron_datasets_split[j])

            # Sum over all contributing datasets, per split
            size_per_split = list(map(sum, zip(*sizes_per_dataset)))

            blended_datasets = []

            for i in range(len(megatron_datasets)):
                is_none = map(lambda _: _ is None, megatron_datasets[i])

                if split[i] == 0.0:
                    assert all(is_none)
                    blended_datasets.append(None)
                else:
                    assert all(is_none) or not any(is_none)
                    blended_datasets.append(
                        self._build_generic_dataset(
                            BlendedDataset,
                            datasets=megatron_datasets[i],
                            weights=weight_per_dataset,
                            size=size_per_split[i],
                            config=self.config,
                        )
                    )

            return blended_datasets

        else:
            blended_datasets = []
            for i in range(len(Split)):
                blend = getattr(self.config, "blend_per_split")[i]

                # Blend is not provided
                if not blend:
                    blended_datasets.append(None)
                    continue

                split_spoof = [0.0] * len(Split)
                split_spoof[i] = 1.0
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend consists of a sigle prefix
                if len(blend) == 1:
                    blended_datasets.append(self._build_megatron_dataset_splits(blend[0], split_spoof, sizes_spoof)[i])

                # Blend consists of multiple weights and prefixes
                else:
                    (
                        prefix_per_dataset,
                        weight_per_dataset,
                        sizes_per_dataset,
                    ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof)

                    megatron_datasets = []
                    for j in range(len(prefix_per_dataset)):
                        megatron_datasets.append(
                            self._build_megatron_dataset_splits(
                                prefix_per_dataset[j],
                                split_spoof,
                                sizes_per_dataset[j],
                            )[i]
                        )

                    size_per_split = list(map(sum, zip(*sizes_per_dataset)))

                    blended_datasets.append(
                        self._build_generic_dataset(
                            BlendedDataset,
                            datasets=megatron_datasets,
                            weights=weight_per_dataset,
                            size=size_per_split[i],
                            config=self.config,
                        )
                    )

            return blended_datasets

    def _build_megatron_dataset_splits(
        self, path_prefix: str, split: list[float], sizes: list[int]
    ) -> list[MegatronDataset | None]:
        """Build each MegatronDataset split from a single MMapIndexedDataset

        Args:
            path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix

            split (list[float]): The dataset split ratios (must sum to 1.00)

            sizes (list[int]): The number of total samples to draw from each split

        Returns:
            list[MegatronDataset | None]: The MegatronDatset (or None) per split
        """
        indexed_dataset = self._get_indexed_dataset(path_prefix)

        if indexed_dataset is not None:
            if self.cls.is_split_by_sequence():
                split_idx_bounds = _get_split_indices(split, indexed_dataset.sequence_lengths.shape[0])
            else:
                split_idx_bounds = _get_split_indices(split, indexed_dataset.document_indices.shape[0] - 1)

            split_indices = [
                numpy.arange(
                    start=split_idx_bounds[i],
                    stop=split_idx_bounds[i + 1],
                    step=1,
                    dtype=_get_appropriate_dtype_for_range(split_idx_bounds),
                )
                for i, _ in enumerate(Split)
            ]
        else:
            split_indices = [None for _ in Split]

        megatron_datasets = []
        for i, _split in enumerate(Split):
            if split[i] == 0.0:
                megatron_datasets.append(None)
            else:
                megatron_datasets.append(
                    self._build_generic_dataset(
                        self.cls,
                        indexed_dataset=indexed_dataset,
                        indexed_indices=split_indices[i],
                        num_samples=sizes[i],
                        index_split=_split,
                        tokenizer=self.tokenizer,
                        config=self.config,
                    )
                )

        return megatron_datasets

    def build_dataset_single_split(
        self,
        group_names: list[list[str]],
        split_paths: list[list[str]],
        split_splits: list[list[str]],
        split_weights: list[list[str]],
        data_split: Split,
    ) -> list[BlendedDataset | MegatronDataset | None]:
        megatron_datasets = []
        data_split_index = data_split.value

        assert len(split_paths) == len(group_names)
        assert len(split_paths) == len(split_splits)
        assert len(split_paths) == len(split_weights)

        for names, paths, splits, weights in zip(group_names, split_paths, split_splits, split_weights):
            assert len(paths) == len(splits)
            assert len(paths) == len(weights)

            if len(paths) == 1:
                assert weights[0] == 1

                dataset = self._build_megatron_dataset_single_split(
                    names[0], paths[0], splits[0], self.sizes[data_split_index], data_split
                )
                megatron_datasets.append(dataset)
            else:
                blend = []
                for w, p in zip(weights, paths):
                    blend.append(w)
                    blend.append(p)

                _, weights, sizes = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)

                datasets = []
                for name, path, split, size in zip(names, paths, splits, sizes):
                    dataset = self._build_megatron_dataset_single_split(
                        name, path, split, size[data_split_index], data_split
                    )
                    datasets.append(dataset)

                size_per_split = list(map(sum, zip(*sizes)))

                megatron_datasets.append(
                    self._build_generic_dataset(
                        BlendedDataset,
                        datasets=datasets,
                        weights=weights,
                        size=size_per_split[data_split_index],
                        config=self.config,
                    )
                )

        return megatron_datasets

    def _build_megatron_dataset_single_split(
        self, group_name: str, path_prefix: str, split: str, size: int, data_split: Split
    ) -> list[MegatronDataset | None]:
        indexed_dataset = self._get_indexed_dataset(path_prefix)

        start, end = _parse_split(split)

        modified_config = deepcopy(self.config)
        modified_config.name = group_name
        modified_config.split = split

        split_indices = None
        if indexed_dataset is not None:
            num_elements = (
                indexed_dataset.sequence_lengths.shape[0]
                if self.cls.is_split_by_sequence()
                else indexed_dataset.document_indices.shape[0] - 1
            )
            start = int(start * num_elements)
            end = int(end * num_elements)

            split_indices = numpy.arange(
                start=start,
                stop=end,
                step=1,
                dtype=_get_appropriate_dtype_for_range([start, end]),
            )

        megatron_dataset = None
        if start != end:
            megatron_dataset = self._build_generic_dataset(
                self.cls,
                indexed_dataset=indexed_dataset,
                indexed_indices=split_indices,
                num_samples=size,
                index_split=data_split,
                tokenizer=self.tokenizer,
                config=modified_config,
            )

        return megatron_dataset

    def _build_generic_dataset(self, cls: type[DistributedDataset], **kwargs: Any) -> DistributedDataset | None:
        """Build the DistributedDataset

        Return None if and only if the underlying MegatronDataset class is not built on the current
        rank and torch.distributed is initialized.

        Args:
            cls (type[DistributedDataset]): The DistributedDataset class to be built

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            DistributedDataset | None: The DistributedDataset instantion or None
        """
        if torch.distributed.is_initialized():
            rank = ProcessGroupManager.get_global_rank()
            caching_allowed = rank == 0 or (torch.cuda.current_device() == 0 and self.config.node_uses_local_storage)

            dataset = None

            # First, build on rank 0
            if caching_allowed and self.config.is_built_on_rank:
                try:
                    dataset = cls(**kwargs, caching_allowed=True)
                except OSError as err:
                    log = (
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            torch.distributed.barrier()

            # After, build on other ranks
            if not caching_allowed and self.config.is_built_on_rank:
                dataset = cls(**kwargs, caching_allowed=False)

            return dataset

        return cls(**kwargs, caching_allowed=True)

    def _get_indexed_dataset(self, path_prefix: str) -> MMapIndexedDataset | None:
        indexed_dataset = None
        if not torch.distributed.is_initialized() or self.config.is_built_on_rank:
            indexed_dataset = MMapIndexedDataset(path_prefix, self.cls.is_multimodal())

        return indexed_dataset


def _get_split_indices(split: list[float], num_elements: int) -> List[int]:
    """Determine the document index bounds per split

    Args:
        split (list[float]): The dataset split ratios (must sum to 1.00)

        num_elements (int): The number of elements, e.g. sequences or documents, available for
        the split

    Returns:
        list[int]: The indices for all three splits e.g. [0, 900, 990, 1000] for a 1000-document
        set and a [90.0, 9.0, 1.0] split
    """
    split_indices = [0]
    for split_pct in split:
        split_indices.append(split_indices[-1] + int(round(split_pct * float(num_elements))))
    split_indices[1:] = list(map(lambda _: _ - (split_indices[-1] - num_elements), split_indices[1:]))

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices


def _get_prefixes_weights_and_sizes_for_blend(
    blend: list[str], target_num_samples_per_split: list[int]
) -> tuple[list[str], list[float], list[list[int]]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits

    Args:
        blend (list[str]): e.g. ["30", "path/to/dataset_1_prefix", "70",
        "path/to/dataset_2_prefix"]

        target_num_samples_per_split (list[int]): The number of samples to target for each
        BlendedDataset split

    Returns:
        tuple[list[str], list[float], list[list[int]]]: The prefix strings e.g.
        ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], the normalized weights e.g.
        [0.3, 0.7], and the number of samples to request per MegatronDataset per split
    """
    weights, prefixes = zip(*[(float(blend[i]), blend[i + 1].strip()) for i in range(0, len(blend), 2)])

    weights = normalize(weights)

    # Use 0.5% target margin to ensure we satiate the network
    sizes_per_dataset = [
        [int(math.ceil(target_num_samples * weight * 1.005)) for target_num_samples in target_num_samples_per_split]
        for weight in weights
    ]

    return prefixes, weights, sizes_per_dataset


def _parse_split(start_end: str) -> tuple[float]:
    start, end = start_end.split(":")
    start = float(start)
    end = float(end)

    return start, end


def _get_appropriate_dtype_for_range(split_idx_bounds: list[int]) -> numpy.dtype:
    max_value = max(split_idx_bounds)

    if max_value <= numpy.iinfo(numpy.int32).max:
        dtype = numpy.int32
    elif max_value <= numpy.iinfo(numpy.int64).max:
        dtype = numpy.int64
    else:
        raise ValueError("value for split idx is too large")

    return dtype
