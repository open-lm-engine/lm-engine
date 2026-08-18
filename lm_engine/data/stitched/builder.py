# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Build (or load cached) the sample_index for StitchedSequenceDataset.

sample_index shape: [N+1, 3]  dtype: int64
Each row encodes a boundary as (collection_idx, doc_offset_within_collection, token_offset_within_doc).
Sample i spans from sample_index[i] to sample_index[i+1].
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import numpy as np
import pandas as pd

from ...logging_utils import log_rank_0
from ...parallel import ProcessGroupManager
from ..megatron import Split
from ..megatron.indexed_dataset import MMapIndexedDataset
from .config import StitchedDatasetConfig

_POLL_INTERVAL_SECONDS = 5


def build_sample_index(
    config: StitchedDatasetConfig,
    split: Split,
    num_samples: int | None,
    caching_allowed: bool,
    _is_builder: bool | None = None,
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
        _is_builder: Test hook to force the builder/waiter role without a distributed
            setup. None (default) means: single-process (torch.distributed not
            initialized) -> builder; otherwise global rank 0 builds and every other
            rank polls for the cache file.

    Returns:
        stitched_seq: Full stitched_seq DataFrame (all collections, all splits).
        sample_index: int64 array of shape [N+1, 3] encoding sample boundaries.
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

        if _is_builder is None:
            is_builder = not ProcessGroupManager.is_initialized() or ProcessGroupManager.get_global_rank() == 0
        else:
            is_builder = _is_builder

        if not is_builder:
            # Only rank 0 builds: all other ranks poll for the atomically-saved cache
            # file instead of duplicating the (potentially minutes-long) build work.
            sample_index = _wait_for_cached_sample_index(cache_path, config.cache_build_timeout_seconds)
            return stitched_seq, sample_index

        if cache_path.is_file():
            log_rank_0(logging.INFO, f"Loading cached sample_index from {cache_path}")
            try:
                sample_index = np.load(cache_path)
                # stitched_seq is the full DataFrame; coll values in sample_index are global
                return stitched_seq, sample_index
            except Exception:
                # A corrupt/partial cache (e.g. left behind by a crashed older run
                # that saved non-atomically) must not kill the job — rebuild instead.
                log_rank_0(
                    logging.WARNING,
                    f"  Cached sample_index at {cache_path} is unreadable; rebuilding and overwriting.",
                )

    log_rank_0(logging.INFO, "Building StitchedSequenceDataset sample_index ...")
    t0 = time.time()

    # Load each unique shard's sequence_lengths array once.
    # Bind `ds` to a local name so the MMapIndexedDataset stays alive through
    # the entire .astype() copy. The chained form
    # `MMapIndexedDataset(...).index.sequence_lengths.astype(...)` segfaults
    # under concurrent multi-rank load (verified on prob-0.5 + prob-0.9 with
    # caching disabled): the temporary dataset's mmap can be released mid-cast
    # when the only reference chain is via memoryview/np.frombuffer.
    shard_lengths_cache: dict[str, np.ndarray] = {}
    for shard_id in split_seq["shard"].unique():
        path_prefix = os.path.join(config.tokenized_data_root, shard_id)
        ds = MMapIndexedDataset(path_prefix)
        shard_lengths_cache[shard_id] = ds.index.sequence_lengths.astype(np.int64)
        del ds

    shards_arr = split_seq["shard"].values
    begs_arr = split_seq["coll_beg"].values.astype(np.int64)
    ends_arr = split_seq["coll_end"].values.astype(np.int64)
    collection_sizes = (ends_arr - begs_arr).astype(np.int64)

    if (collection_sizes == 1).all():
        # Fast path: every collection is a single document — vectorized fancy indexing per shard
        all_lengths = np.empty(len(split_seq), dtype=np.int64)
        for shard_id, shard_seq_lengths in shard_lengths_cache.items():
            mask = shards_arr == shard_id
            all_lengths[mask] = shard_seq_lengths[begs_arr[mask]]
    else:
        # General path: collections span multiple documents
        per_collection_lengths = [
            shard_lengths_cache[row.shard][row.coll_beg : row.coll_end] for row in split_seq.itertuples(index=False)
        ]
        all_lengths = np.concatenate(per_collection_lengths)

    log_rank_0(logging.INFO, f"  Loaded token lengths for {len(split_seq)} collections ({time.time()-t0:.1f}s)")

    # Phase 1: build doc-level prefix sums

    # collection_starts[i] = total docs before collection i (used to map global doc index -> collection)
    collection_starts = np.zeros(len(collection_sizes) + 1, dtype=np.int64)
    collection_starts[1:] = np.cumsum(collection_sizes)

    # all_lengths[i] = token count of global doc i
    total_docs = int(collection_starts[-1])

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
    sample_index = np.stack([coll_indices, doc_offsets_within_coll, token_offsets], axis=1).astype(np.int64)

    log_rank_0(logging.INFO, f"  Built sample_index shape={sample_index.shape} in {time.time()-t0:.1f}s")

    if caching_allowed:
        _atomic_save(cache_path, sample_index)
        log_rank_0(logging.INFO, f"  Saved sample_index to {cache_path}")

    return stitched_seq, sample_index


def _wait_for_cached_sample_index(cache_path: os.PathLike, timeout_seconds: float) -> np.ndarray:
    """Poll until the rank-0-built cache appears and loads successfully.

    The cache is saved atomically (temp file + os.replace), so a path that shows
    up in is_file() always contains a complete file — except for torn files left
    behind by crashed older runs that saved non-atomically. A load failure within
    the timeout therefore means rank 0 is about to atomically overwrite a stale
    corrupt cache; keep polling instead of giving up. Raises TimeoutError if the
    deadline passes without a successful load (e.g. the builder rank crashed, or
    this code path is never reached on rank 0).
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if cache_path.is_file():
            try:
                return np.load(cache_path)
            except Exception:
                pass
        time.sleep(_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for rank 0 to build the sample_index cache "
        f"at {cache_path}. Check the rank 0 logs for build errors, or increase "
        f"StitchedDatasetConfig.cache_build_timeout_seconds."
    )


def _atomic_save(cache_path: os.PathLike, sample_index: np.ndarray) -> None:
    """Save sample_index to cache_path atomically.

    Non-builder ranks poll for this file on a shared filesystem, and independent
    jobs can share a cache directory. A plain np.save() truncates the target in
    place, so a reader that passes the is_file() check while a writer is mid-save
    reads a torn file (EOFError / "contains pickled (object) data" ValueError).
    Writing to a unique temp file in the same directory and os.replace()-ing it
    into place makes the file appear atomically: readers see either the old
    complete file or the new complete file, never a partial one. The build is
    deterministic, so concurrent writers produce identical content and
    last-writer-wins is safe.
    """
    cache_path = str(cache_path)
    tmp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp"
    try:
        # np.save() appends .npy to paths without the extension; pass an open
        # file handle to keep full control of the temp filename.
        with open(tmp_path, "wb") as tmp_file:
            np.save(tmp_file, sample_index)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
