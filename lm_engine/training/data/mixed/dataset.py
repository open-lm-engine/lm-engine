# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""MixedSequenceDataset: mixture over multiple pretraining datasets.

Each child dataset contributes some fixed-length samples (each a 1-D int64 array/tensor of
sequence_length + 1 tokens under the "text" key; remainders dropped). One mixture epoch visits every
sample of every child exactly once, in a deterministic seeded shuffle -- equivalent to concatenating
all child datasets and shuffling. When more than one mixture epoch is requested, the later epochs
are replayed with a fresh per-epoch shuffle (if reshuffle_per_epoch is True) or in the same
deterministic order (if reshuffle_per_epoch is False, used for val/test sets).

With a single child dataset there is nothing to mix: samples are served in exactly that child's
own order, with no shuffling applied by this layer.

The global index -> (child, local index) mapping is a function of the children sizes and the seed,
so resume via consumed_samples needs no dataloader state.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from ...logging_utils import log_rank_0


class MixedSequenceDataset(torch.utils.data.Dataset):
    """Mixture of multiple child datasets.

    Args:
        datasets: Child datasets; each __getitem__ must return a dict whose "text" entry holds
            sequence_length + 1 tokens (numpy array or tensor).
        data_names: Name per child, used for logging.
        num_samples: Total samples to serve; may exceed one mixture epoch, in which case epochs are
            replayed with a per-epoch reshuffle (if reshuffle_per_epoch is True) or in the same
            order (if False).
        seed: Seed for the per-epoch permutations.
        reshuffle_per_epoch: If True, reshuffle data on each epoch boundary (for training). If
            False, replay data in the same deterministic order (for val/test sets). Default True.
    """

    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
        data_names: list[str],
        num_samples: int,
        seed: int,
        reshuffle_per_epoch: bool = True,
    ) -> None:
        assert len(datasets) > 0, f"{self.__class__.__name__} must have at least one child dataset"
        assert len(datasets) == len(data_names), f"{self.__class__.__name__} must have a name for each child dataset"

        self.datasets = datasets
        self._num_samples = int(num_samples)
        self._seed = seed
        self._reshuffle_per_epoch = reshuffle_per_epoch

        epoch_sizes = [len(dataset) for dataset in datasets]
        assert all(size > 0 for size in epoch_sizes), (
            f"{self.__class__.__name__} every source needs a non-empty train split, " f"got epoch sizes: {epoch_sizes}"
        )

        # offsets[i] is the first pre-shuffle position of child i within a mixture epoch;
        # the last entry is the entire mixture epoch size, preserved for sanity checks.
        self._offsets = np.cumsum([0] + epoch_sizes)
        self._epoch_samples = int(self._offsets[-1])

        for data_name, size in zip(data_names, epoch_sizes):
            log_rank_0(
                logging.INFO,
                f"{self.__class__.__name__}: source {data_name}: {size} samples/epoch "
                f"({size / self._epoch_samples:.2%} of the mixture)",
            )
        if self._num_samples > self._epoch_samples:
            replay_msg = (
                "data replayed with a per-epoch reshuffle"
                if self._reshuffle_per_epoch
                else "data replayed in same order"
            )
            log_rank_0(
                logging.INFO,
                f"{self.__class__.__name__}: "
                f"serving {self._num_samples} samples over {self._epoch_samples} "
                f"samples/epoch (~{math.ceil(self._num_samples / self._epoch_samples)} epochs; "
                f"{replay_msg})",
            )

        # Per-epoch sample index permutations, built lazily and cached in _get_permutation;
        # Dropped on pickling and re-computed in workers.
        self._permutations: dict[int, np.ndarray] = {}

        # Single source fallbacks to exact StitchDataLoader behavior
        self._is_single_source = len(self.datasets) == 1

    def __len__(self) -> int:
        return self._num_samples

    def _get_permutation(self, epoch_idx: int) -> np.ndarray:
        """Get the permutation for a given mixture epoch.

        With a single child dataset there is nothing to mix: returns the identity permutation
        (cached under epoch 0, since it is the same for every epoch) so samples are served in
        exactly that child's own order, unshuffled by this layer, on every replayed epoch --
        regardless of seed.

        If reshuffle_per_epoch is False, always returns epoch 0's permutation
        for deterministic replay across all epochs (used for val/test sets).

        Args:
            epoch_idx: Mixture epoch index (0-based).

        Returns:
            An np.ndarray: permutation of indices 0..._epoch_samples for the given mixture epoch.
        """
        actual_epoch = 0 if not self._reshuffle_per_epoch else epoch_idx

        perm = self._permutations.get(actual_epoch)
        if perm is None:
            perm = (
                np.arange(self._epoch_samples)
                if self._is_single_source
                else np.random.default_rng([self._seed, actual_epoch]).permutation(self._epoch_samples)
            )
            self._permutations[actual_epoch] = perm
        return perm

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the mixture.

        First map the global sample index to a mixture epoch index and sample index within that
        epoch, then map that index to a child dataset and a local index within that child.
        Finally, fetch the sample from the child dataset and return the normalized entry.

        Args:
            idx: Global sample index.

        Returns:
            A dictionary containing the sample data.
        """
        if self._is_single_source:
            sample = self.datasets[0][idx]
        else:
            epoch_idx, idx_within_epoch = divmod(int(idx), self._epoch_samples)
            position = int(self._get_permutation(epoch_idx)[idx_within_epoch])
            dataset_id = int(np.searchsorted(self._offsets, position, side="right")) - 1
            sample = self.datasets[dataset_id][position - int(self._offsets[dataset_id])]

            # Children return tensors (StitchedSequenceDataset), but normalize defensively
            # (dtype + drop any extra keys) so batches collate uniformly.
        return {"text": torch.as_tensor(sample["text"], dtype=torch.int64)}

    def state_dict(self) -> dict:
        """Returns an empty dict as all construction arguments come from metadata."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """No-op; all construction arguments come from metadata."""

    def __getstate__(self) -> dict:
        """Drop the cached permutations on pickling, which will be re-computed in workers.

        Returns:
            A dictionary representing the state of the object, excluding cached permutations.
        """
        state = self.__dict__.copy()
        state["_permutations"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the object from the given state dictionary."""
        self.__dict__.update(state)
