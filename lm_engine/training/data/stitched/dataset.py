# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""StitchedSequenceDataset: produces fixed-length token sequences from a
stitched collection index.

Each training sample is a 1-D int64 tensor of length (sequence_length + 1).
The +1 is the label shift: input = tokens[:-1], labels = tokens[1:], which
is handled downstream by the collate_fn.

__getitem__(i):
  - Looks up sample_index[i] and sample_index[i+1] to get boundary coords
    (c0, d0, t0) and (c1, d1, t1).
  - Walks collections c0..c1 from the stitched_seq DataFrame.
  - For each collection, applies the ordering strategy to get doc visit order.
  - Reads tokens from the ShardStore, slicing the first and last doc as needed.
  - Concatenates into a seq_len+1 token array.

The DataLoader is responsible for batching; this dataset returns one sample at a
time (micro-batch assembly is done by MegatronBatchSampler, same as GPTDataset).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch

from ...logging_utils import log_rank_0
from ..megatron import Split
from .config import OrderingStrategy, StitchedDatasetConfig
from .ordering import get_doc_order
from .shard_store import ShardStore


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """Dataset backed by a stitched sequence parquet file.

    Args:
        config: StitchedDatasetConfig instance.
        split: Which split (train / val / test).
        stitched_seq: Full DataFrame [shard, coll_beg, coll_end] (all collections); c-values in sample_index are global row indices into this.
        sample_index: int64 array of shape [N+1, 3] encoding one full pass (epoch) over the split.
        num_samples: Total samples to serve. If it exceeds one epoch, the epoch is
            replayed with a per-epoch reshuffle. None serves exactly one epoch.
    """

    def __init__(
        self,
        config: StitchedDatasetConfig,
        split: Split,
        stitched_seq: pd.DataFrame,
        sample_index: np.ndarray,
        num_samples: int | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.stitched_seq = stitched_seq
        self.sample_index = sample_index  # [N+1, 3] int64

        assert sample_index.ndim == 2 and sample_index.shape[1] == 3
        self._n_epoch_samples = len(sample_index) - 1  # samples in one pass over the split
        assert self._n_epoch_samples > 0, "sample_index must encode at least one sample"

        self._num_samples = self._n_epoch_samples if num_samples is None else int(num_samples)
        if self._num_samples > self._n_epoch_samples:
            log_rank_0(
                logging.INFO,
                f"StitchedSequenceDataset[{split.name}]: serving {self._num_samples} samples over "
                f"{self._n_epoch_samples}/epoch (~{math.ceil(self._num_samples / self._n_epoch_samples)} "
                f"epochs, replayed with a per-epoch reshuffle)",
            )

        # ShardStore is created here (in the main process) but is fully
        # re-initialised in each DataLoader worker via __getstate__/__setstate__.
        self._shard_store = ShardStore(config.tokenized_data_root)

        # RNG for random ordering; seeded per-sample inside __getitem__ for
        # reproducibility across workers and restarts.
        self._base_seed = config.seed

        # Per-epoch permutations, built lazily. Each epoch reshuffles independently
        # so replays are reordered; the on-disk sample_index is untouched.
        self._permutations: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._num_samples

    def _get_permutation(self, epoch: int) -> np.ndarray:
        perm = self._permutations.get(epoch)
        if perm is None:
            perm = np.random.default_rng([self._base_seed, self.split.value, epoch]).permutation(self._n_epoch_samples)
            self._permutations[epoch] = perm
        return perm

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # idx -> (epoch, position), then through that epoch's permutation to a row.
        epoch, base = divmod(int(idx), self._n_epoch_samples)
        true_idx = int(self._get_permutation(epoch)[base])
        tokens = self._fetch_tokens(true_idx)
        return {"text": torch.from_numpy(tokens.astype(np.int64))}

    def _fetch_tokens(self, idx: int) -> np.ndarray:
        seq_len = self.config.sequence_length
        need = seq_len + 1  # +1 for the label shift

        c0, d0, t0 = [int(x) for x in self.sample_index[idx]]
        c1, d1, t1 = [int(x) for x in self.sample_index[idx + 1]]

        parts: list[np.ndarray] = []

        for c in range(c0, c1 + 1):
            row = self.stitched_seq.iloc[c]
            shard_id = row["shard"]
            beg = int(row["coll_beg"])
            end = int(row["coll_end"])

            # Always traverse in as_stored order so d0/d1/t0/t1 from sample_index
            # (computed assuming as_stored) correctly index into local_docs.
            as_stored = get_doc_order(beg, end, OrderingStrategy.as_stored)

            if c == c0 and c == c1:
                # Sample entirely within one collection
                local_docs = as_stored[d0 : d1 + 1]
            elif c == c0:
                # First collection: start at d0
                local_docs = as_stored[d0:]
            elif c == c1:
                # Last collection: end at d1
                local_docs = as_stored[: d1 + 1]
            else:
                # Middle collection: take all docs
                local_docs = as_stored

            # Read chunks in as_stored order into tmp
            tmp: list[np.ndarray] = []
            for j, doc_idx in enumerate(local_docs):
                doc_idx = int(doc_idx)
                is_first_doc = c == c0 and j == 0
                is_last_doc = c == c1 and j == len(local_docs) - 1

                if is_first_doc and is_last_doc:
                    # Read [t0, t1] inclusive — t1 is the last label token of this sample
                    length = t1 - t0 + 1
                    chunk = self._shard_store.get_tokens(shard_id, doc_idx, offset=t0, length=length)
                elif is_first_doc:
                    # Read from t0 to end of doc
                    chunk = self._shard_store.get_tokens(shard_id, doc_idx, offset=t0)
                elif is_last_doc:
                    # Read [0, t1] inclusive — t1 is the last label token of this sample
                    chunk = self._shard_store.get_tokens(shard_id, doc_idx, offset=0, length=t1 + 1)
                else:
                    # Full document
                    chunk = self._shard_store.get_tokens(shard_id, doc_idx)

                tmp.append(chunk)

            # Reorder chunks within this collection according to ordering strategy
            rng = np.random.RandomState((self._base_seed + idx * 31337 + c) % (2**32))
            order = get_doc_order(0, len(tmp), self.config.ordering_strategy, rng=rng)
            parts.extend(tmp[i] for i in order)

        tokens = np.concatenate(parts)
        # The clamped (short) sample, if any, is the last row of the epoch.
        is_last_sample = idx == self._n_epoch_samples - 1
        if is_last_sample:
            assert len(tokens) <= need, (
                f"sample {idx} (last): got {len(tokens)} tokens, expected <= {need}. "
                "This indicates a bug in build_sample_index."
            )
        else:
            assert len(tokens) == need, (
                f"sample {idx}: got {len(tokens)} tokens, expected {need}. "
                "This indicates a bug in build_sample_index."
            )

        return tokens

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_shard_store"] = None  # re-created in the worker
        state["_permutations"] = {}  # recomputed deterministically per worker
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._shard_store = ShardStore(self.config.tokenized_data_root)
