# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import re
from dataclasses import dataclass, field

from ...utils import ProcessGroupManager, log_rank_0
from .utils import Split, normalize


@dataclass
class BlendedMegatronDatasetConfig:
    """Configuration object for megatron-core blended and megatron datasets

    Attributes:
        is_built_on_rank (bool): True if the dataset should be built
        on the current rank. It should be Megatron Core parallelism aware i.e. global rank, group
        rank, and virtual rank may inform its return value.

        random_seed (int): The seed for all RNG during dataset creation.

        sequence_length (int): The sequence length.

        blend (list[str] | None): The blend string, consisting of either a single dataset or a
        flattened sequential sequence of weight-dataset pairs. For example, ["dataset-path1"] and
        ["50", "dataset-path1", "50", "dataset-path2"] are both valid. Not to be used with
        'blend_per_split'. Defaults to None.

        blend_per_split (blend_per_split: list[list[str] | None]): A set of blend
        strings, as defined above, one for each split distribution. Not to be used with 'blend'.
        Defauls to None.

        split (str | None): The split string, a comma separated weighting for the dataset splits
        when drawing samples from a single distribution. Not to be used with 'blend_per_split'.
        Defaults to None.

        split_vector: (list[float] | None): The split string, parsed and normalized post-
        initialization. Not to be passed to the constructor.

        path_to_cache (str): Where all re-useable dataset indices are to be cached.
    """

    is_built_on_rank: bool

    random_seed: int

    sequence_length: int

    name: str | None = None

    blend: list[str] | None = None

    blend_per_split: list[list[str] | None] | None = None

    split: str | None = None

    split_vector: list[float] | None = field(init=False, default=None)

    path_to_cache: str = None

    node_uses_local_storage: bool = False

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization. See
        https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """

        if ProcessGroupManager.get_global_rank() == 0:
            assert self.is_built_on_rank, "is_built_on_rank must be True when global rank = 0"

        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.blend is None, "blend and blend_per_split are incompatible"
            assert len(self.blend_per_split) == len(Split), f"blend_per_split must contain {len(Split)} blends"
            if self.split is not None:
                self.split = None
                log_rank_0(logging.WARNING, f"Let split = {self.split}")
        elif self.blend is not None:
            assert self.split is not None, "both blend and split must be provided"
            self.split_vector = _parse_and_normalize_split(self.split)
            log_rank_0(logging.INFO, f"Let split_vector = {self.split_vector}")


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for megatron-core blended and megatron GPT datasets

    Attributes:
        return_document_ids (bool): Whether to return the document ids when querying the dataset.
        fim_rate (float): Fill-in-the-middle objective percentage
        fim_spm_rate (float): Probability that the a FIM sample uses the SPM format over the PSM format.
    """

    return_document_ids: bool = False
    fim_rate: float = 0
    fim_spm_rate: float = 0


def _parse_and_normalize_split(split: str) -> list[float]:
    """Parse the dataset split ratios from a string

    Args:
        split (str): The train valid test split string e.g. "99,1,0"

    Returns:
        list[float]: The trian valid test split ratios e.g. [99.0, 1.0, 0.0]
    """
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(Split)
    assert all(map(lambda _: _ >= 0.0, split))

    split = normalize(split)

    return split
