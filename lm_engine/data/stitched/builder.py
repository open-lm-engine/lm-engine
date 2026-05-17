# **************************************************
# Copyright (c) 2026, Jyo Pari
# **************************************************

"""Build (or load cached) the sample_index for StitchedSequenceDataset.

sample_index shape: [N+1, 3]  dtype: int32
Each row encodes a boundary as (collection_idx, doc_offset_within_collection, token_offset_within_doc).
Sample i spans from sample_index[i] to sample_index[i+1].
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ...utils import log_rank_0
from ..megatron import Split
from ..megatron.indexed_dataset import MMapIndexedDataset
from .config import StitchedDatasetConfig


def build_sample_index(
    config: StitchedDatasetConfig,
    split: Split,
    num_samples: int,
    caching_allowed: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (or load from cache) the sample_index for a given split.

    Sample i spans from sample_index[i] to sample_index[i+1], where each row is
    (collection_idx, doc_offset_within_collection, token_offset_within_doc).
    collection_idx is a *global* index into the full stitched_seq DataFrame so
    the caller can resolve shard/coll_beg/coll_end without knowing split boundaries.

    Args:
        config: Dataset configuration (paths, sequence_length, split_ratio, etc.).
        split: Which split to build (train / val / test).
        num_samples: Number of samples to pack. If None, uses the maximum that fits.
        caching_allowed: Whether to read/write the on-disk cache.

    Returns:
        stitched_seq: Full stitched_seq DataFrame (all collections, all splits).
        sample_index: int32 array of shape [N+1, 3] encoding sample boundaries.
    """

    stitched_seq = pd.read_parquet(config.stitched_seq_path)
    assert set(stitched_seq.columns) >= {
        "shard",
        "coll_beg",
        "coll_end",
    }, f"stitched_seq must have columns [shard, coll_beg, coll_end], got {list(stitched_seq.columns)}"

    total_collections = len(stitched_seq)
    split_bounds = _get_split_bounds(total_collections, config.split_ratio)
    lo, hi = split_bounds[split.value], split_bounds[split.value + 1]
    split_seq = stitched_seq.iloc[lo:hi]  # used only for building — not returned
    # TODO: splitting on collections is a design choice worth revisiting.
    # Splitting on generated samples (i.e., after building the full sample_index)
    # would give more precise train/val/test sizes. Currently, split_bounds are
    # also recomputed independently for each split call, which is wasteful

    if len(split_seq) == 0:
        raise ValueError(f"Split {split} has 0 collections after applying split_ratio {config.split_ratio}")

    if caching_allowed:
        cache_dir = config.cache_dir
        cache_path = cache_dir / f"{split.name}_n{num_samples}.npy"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.is_file():
            log_rank_0(logging.INFO, f"Loading cached sample_index from {cache_path}")
            sample_index = np.load(cache_path)
            return stitched_seq, sample_index  # stitched_seq is full DataFrame; c values in sample_index are global

    log_rank_0(logging.INFO, "Building StitchedSequenceDataset sample_index ...")
    t0 = time.time()

    per_collection_lengths: list[np.ndarray] = []
    shard_cache: dict[str, MMapIndexedDataset] = {}

    for _, row in split_seq.iterrows():
        shard_id = row["shard"]
        beg = int(row["coll_beg"])
        end = int(row["coll_end"])

        if shard_id not in shard_cache:
            path_prefix = os.path.join(config.tokenized_data_root, shard_id)
            shard_cache[shard_id] = MMapIndexedDataset(path_prefix)

        lengths = shard_cache[shard_id].index.sequence_lengths[beg:end].astype(np.int64)
        per_collection_lengths.append(lengths)

    log_rank_0(logging.INFO, f"  Loaded token lengths for {len(split_seq)} collections ({time.time()-t0:.1f}s)")

    # Phase 1: build doc-level prefix sums

    # collection_starts[i] = total docs before collection i (used to map global doc index -> collection)
    collection_sizes = np.array([len(x) for x in per_collection_lengths], dtype=np.int64)
    collection_starts = np.zeros(len(collection_sizes) + 1, dtype=np.int64)
    collection_starts[1:] = np.cumsum(collection_sizes)

    # all_lengths[i] = token count of global doc i
    total_docs = int(collection_starts[-1])
    all_lengths = np.concatenate(per_collection_lengths)  # shape [total_docs]

    # Phase 2: build token-level prefix sums

    # global_token_start[j] = total tokens before doc j
    # treating the entire split as one flat token stream
    global_token_start = np.zeros(total_docs + 1, dtype=np.int64)
    global_token_start[1:] = np.cumsum(all_lengths)
    total_tokens = int(global_token_start[-1])

    # Phase 3: compute sample boundaries

    seq_len = config.sequence_length
    # -1 because each sample needs seq_len+1 tokens (input + label shift).
    # e.g., total_tokens=4, seq_len=2: sample 0 needs T0..T2, sample 1 needs T2..T4
    # but T4 doesn't exist, so max = (4-1)//2 = 1, not 4//2 = 2.
    max_possible_samples = max(1, (total_tokens - 1) // seq_len)

    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    n_samples = min(num_samples, max_possible_samples) if num_samples is not None else max_possible_samples
    if num_samples is not None and num_samples > max_possible_samples:
        log_rank_0(
            logging.WARNING,
            f"  Requested {num_samples} samples but only {max_possible_samples} are possible; capping.",
        )

    requested_str = (
        f"{num_samples} (capped at {n_samples})"
        if num_samples is not None and num_samples > max_possible_samples
        else str(n_samples)
    )
    log_rank_0(
        logging.INFO,
        f"  total_docs={total_docs}, total_tokens={total_tokens}, "
        f"max_possible_samples={max_possible_samples}, requested={requested_str}",
    )

    # n_samples+1 boundaries at [0, seq_len, 2*seq_len, ...].
    # Boundaries are *inclusive*: boundary_positions[i] is the token index of the
    # first input token of sample i, and boundary_positions[i+1] is the last label
    # token of sample i (= first input token of sample i+1). This overlap is required
    # for the label shift: input = tokens[:-1], labels = tokens[1:].
    boundary_positions = np.arange(n_samples + 1, dtype=np.int64) * seq_len

    # Clamp so every boundary is a valid token index. Needed when n_samples * seq_len
    # >= total_tokens (e.g. exact divisibility): the last boundary would otherwise
    # point past the end of the token stream.
    boundary_positions = np.minimum(boundary_positions, total_tokens - 1)

    # --- Phase 4: map each boundary position → which doc it lands in and token offset ---

    # doc_indices[i] = the index of the doc that contains the i-th boundary token.
    # searchsorted(..., side="right") - 1 gives the last doc that starts at or before p.
    # e.g., global_token_start = [0, 5, 9, 14]
    #       searchsorted([0,5,9,14,...], 7, side="right") = 2; 2 - 1 = 1
    #       searchsorted([0,5,9,14,...], 5, side="right") = 2; 2 - 1 = 1
    doc_indices = np.searchsorted(global_token_start, boundary_positions, side="right") - 1
    # Defensive clamp — shouldn't trigger given the boundary clamp above,
    # but guards against floating-point edge cases.
    doc_indices = np.clip(doc_indices, 0, total_docs - 1)

    # token_offsets[i] = the 0-indexed token offset within the doc that contains the i-th boundary token.
    # e.g., global_token_start = [0, 5, 9, 14], boundary_positions[i] = 7
    #       token_offsets[i] = boundary_positions[i] - global_token_start[doc_indices[i]]
    #                        = 7 - 5 = 2
    token_offsets = boundary_positions - global_token_start[doc_indices]

    # --- Phase 5: map global doc index → (collection, doc-within-collection) ---
    # which collection the doc belongs to, and doc's index/offset within the collection.

    # coll_indices[i] = the 0-indexed index of the collection inside stitched seq
    # e.g., collection_starts = [0, 3, 7, 10] and doc_indices[i] = 5
    #       searchsorted([0,3,7,10], 5, side="right") → 2; 2 - 1 = 1
    #       doc_offsets_within_coll = 5 - collection_starts[1] = 5 - 3 = 2
    # Same searchsorted trick over collection_starts to find which collection each doc belongs to.
    coll_indices = np.searchsorted(collection_starts, doc_indices, side="right") - 1
    coll_indices = np.clip(coll_indices, 0, len(split_seq) - 1)
    # Doc's local index within its collection.
    doc_offsets_within_coll = doc_indices - collection_starts[coll_indices]
    # coll_indices are 0-indexed within the split; add lo to get the row index in stitched_seq.
    coll_indices = coll_indices + lo

    # --- Phase 6: pack into sample_index ---

    # Shape [n_samples+1, 3]. Each row encodes a boundary token as
    # (coll_idx in stitched_seq, doc_offset_within_coll, token_offset_within_doc).
    #
    # To read sample i: start at boundary i (coll_idx, doc_offset, token_offset),
    # walk forward through docs and collections in order, stop at boundary i+1.
    # Boundary i+1 is the last label token of sample i AND the first input token
    # of sample i+1 — the same token, shared due to the label shift (input=tokens[:-1],
    # labels=tokens[1:]).
    sample_index = np.stack([coll_indices, doc_offsets_within_coll, token_offsets], axis=1).astype(np.int32)

    log_rank_0(logging.INFO, f"  Built sample_index shape={sample_index.shape} in {time.time()-t0:.1f}s")

    if caching_allowed:
        np.save(str(cache_path), sample_index)
        log_rank_0(logging.INFO, f"  Saved sample_index to {cache_path}")

    return stitched_seq, sample_index


def _get_split_bounds(total: int, split_ratio: tuple[float, float, float]) -> list[int]:
    """Return [0, train_end, val_end, total] collection indices.

    Boundaries are computed by accumulating fractions and rounding at each
    split point; the last boundary is always exactly `total` to avoid float
    drift. Raises ValueError if any non-zero split gets 0 collections after
    rounding (total is too small for the given ratio).
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, f"split_ratio must sum to 1.0, got {split_ratio}"
    bounds = [0]
    cumulative = 0
    for frac in split_ratio[:-1]:
        cumulative += frac
        bounds.append(round(cumulative * total))
    bounds.append(total)
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        if lo == hi and split_ratio[i] > 0.0:
            raise ValueError(
                f"Split {i} is empty (bounds={bounds}). "
                f"Not enough collections ({total}) for the given split_ratio."
            )
    return bounds
