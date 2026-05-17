# **************************************************
# Copyright (c) 2026, Jyo Pari, Mayank Mishra
# **************************************************

"""Unit tests for StitchedSequenceDataset.

These tests are self-contained: they build small synthetic MMapIndexedDatasets
and a matching stitched_seq parquet file in a temp directory, then exercise
build_sample_index and StitchedSequenceDataset.__getitem__.

No GPUs, no distributed state required.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from lm_engine.data.megatron import Split
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDatasetBuilder, get_bin_path, get_idx_path
from lm_engine.data.stitched.builder import build_sample_index
from lm_engine.data.stitched.config import OrderingStrategy, StitchedDatasetConfig
from lm_engine.data.stitched.dataset import StitchedSequenceDataset


def _make_shard(shard_dir: str, stem: str, docs: list[list[int]]) -> str:
    """Write a MMapIndexedDataset shard and return its path_prefix."""
    os.makedirs(shard_dir, exist_ok=True)
    prefix = os.path.join(shard_dir, stem)
    builder = MMapIndexedDatasetBuilder(get_bin_path(prefix))
    for doc in docs:
        builder.add_item(torch.tensor(doc, dtype=torch.int32))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))
    return prefix


def _make_stitched_seq(path: str, records: list[dict]) -> None:
    """Write a stitched_seq parquet file."""
    df = pd.DataFrame(records, columns=["shard", "coll_beg", "coll_end"])
    df["coll_beg"] = df["coll_beg"].astype(np.int32)
    df["coll_end"] = df["coll_end"].astype(np.int32)
    df.to_parquet(path, index=False)


def _build_dataset(
    config: StitchedDatasetConfig,
    split: Split,
    num_samples: int,
    caching_allowed: bool = False,
) -> StitchedSequenceDataset:
    stitched_seq, sample_index = build_sample_index(config, split, num_samples, caching_allowed)
    return StitchedSequenceDataset(config, split, stitched_seq, sample_index)


@pytest.fixture()
def simple_env(tmp_path):
    """
    Two shards, each with 10 documents of length 5.
    Tokens in shard A doc d: [d*10 .. d*10+4]
    Tokens in shard B doc d: [100 + d*10 .. 100 + d*10+4]

    stitched_seq: 4 collections
      col 0: shard_a, docs [0, 3)   (3 docs, 15 tokens)
      col 1: shard_a, docs [3, 7)   (4 docs, 20 tokens)
      col 2: shard_b, docs [0, 5)   (5 docs, 25 tokens)
      col 3: shard_b, docs [5, 10)  (5 docs, 25 tokens)
    Total: 85 tokens across 4 collections.
    """
    data_root = str(tmp_path / "data")

    # shard_a: 10 docs, each doc = [d*10, d*10+1, d*10+2, d*10+3, d*10+4]
    shard_a_docs = [[d * 10 + i for i in range(5)] for d in range(10)]
    _make_shard(os.path.join(data_root, "shard_a"), "shard_a_content", shard_a_docs)

    # shard_b: 10 docs, each doc = [100+d*10, ..., 100+d*10+4]
    shard_b_docs = [[100 + d * 10 + i for i in range(5)] for d in range(10)]
    _make_shard(os.path.join(data_root, "shard_b"), "shard_b_content", shard_b_docs)

    stitched_seq_path = str(tmp_path / "stitched_seq.parquet")
    _make_stitched_seq(
        stitched_seq_path,
        [
            {"shard": "shard_a/shard_a_content", "coll_beg": 0, "coll_end": 3},
            {"shard": "shard_a/shard_a_content", "coll_beg": 3, "coll_end": 7},
            {"shard": "shard_b/shard_b_content", "coll_beg": 0, "coll_end": 5},
            {"shard": "shard_b/shard_b_content", "coll_beg": 5, "coll_end": 10},
        ],
    )

    return {
        "data_root": data_root,
        "stitched_seq_path": stitched_seq_path,
        "shard_a_docs": shard_a_docs,
        "shard_b_docs": shard_b_docs,
        "total_tokens": 85,
    }


def test_sample_index_shape(simple_env):
    """sample_index should be [num_samples+1, 3]."""
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=10,
        split_ratio=(1.0, 0.0, 0.0),
    )
    num_samples = 5
    _, sample_index = build_sample_index(config, Split.train, num_samples, caching_allowed=False)

    assert sample_index.shape == (num_samples + 1, 3), sample_index.shape
    assert sample_index.dtype == np.int32


def test_sample_index_boundaries_are_seq_len_apart(simple_env):
    """Consecutive boundaries should be exactly seq_len tokens apart (modulo edge)."""
    seq_len = 7
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    total_tokens = simple_env["total_tokens"]  # 85
    num_samples = (total_tokens - 1) // seq_len  # 12
    _, sample_index = build_sample_index(config, Split.train, num_samples, caching_allowed=False)

    ds = StitchedSequenceDataset(
        config,
        Split.train,
        pd.read_parquet(simple_env["stitched_seq_path"]),
        sample_index,
    )
    for i in range(len(ds)):
        item = ds[i]
        assert item["text"].shape == (seq_len + 1,), f"sample {i} has wrong length {item['text'].shape}"


def test_sample_index_caching(simple_env):
    """Second call should load from cache (no rebuild)."""
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=10,
        split_ratio=(1.0, 0.0, 0.0),
    )
    _, idx1 = build_sample_index(config, Split.train, 5, caching_allowed=True)
    cache_dir = config.cache_dir
    assert cache_dir.exists(), "cache dir should have been created"
    assert len(list(cache_dir.iterdir())) == 1, "expected exactly one cached file"

    _, idx2 = build_sample_index(config, Split.train, 5, caching_allowed=True)
    np.testing.assert_array_equal(idx1, idx2)


def test_split_ratio_divides_collections(simple_env):
    """Train/val split should give disjoint collections."""
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=5,
        split_ratio=(0.5, 0.5, 0.0),
    )
    train_seq, train_si = build_sample_index(config, Split.train, 3, caching_allowed=False)
    val_seq, val_si = build_sample_index(config, Split.valid, 2, caching_allowed=False)

    # Both return the full 4-collection DataFrame
    assert len(train_seq) == 4
    assert len(val_seq) == 4

    # Train c values are in [0, 1], val c values are in [2, 3]
    assert train_si[:, 0].max() <= 1
    assert val_si[:, 0].min() >= 2


def test_getitem_returns_correct_shape(simple_env):
    seq_len = 8
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    ds = _build_dataset(config, Split.train, num_samples=5)

    for i in range(len(ds)):
        item = ds[i]
        assert "text" in item
        assert isinstance(item["text"], torch.Tensor)
        assert item["text"].shape == (seq_len + 1,)
        assert item["text"].dtype == torch.int64


def test_getitem_tokens_are_from_known_docs(simple_env):
    """All tokens in a sample must come from documents we wrote."""
    seq_len = 10
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    ds = _build_dataset(config, Split.train, num_samples=6)

    # Build the full set of valid token values
    all_valid = set()
    for doc in simple_env["shard_a_docs"] + simple_env["shard_b_docs"]:
        all_valid.update(doc)

    for i in range(len(ds)):
        tokens = ds[i]["text"].tolist()
        for tok in tokens:
            assert tok in all_valid, f"unexpected token {tok} in sample {i}"


def test_as_stored_vs_shuffled_ordering_differ(simple_env):
    """shuffled ordering should (very likely) produce different token order than as_stored."""
    seq_len = 20
    # Both use the same sample_index (ordering strategy doesn't affect sample_index)
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config, Split.train, 3, caching_allowed=False)

    config_as_stored = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    config_shuffled = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.shuffled,
        split_ratio=(1.0, 0.0, 0.0),
    )

    ds_as_stored = StitchedSequenceDataset(config_as_stored, Split.train, stitched_seq, sample_index)
    ds_shuffled = StitchedSequenceDataset(config_shuffled, Split.train, stitched_seq, sample_index)

    any_different = False
    for i in range(min(len(ds_as_stored), len(ds_shuffled))):
        if not torch.equal(ds_as_stored[i]["text"], ds_shuffled[i]["text"]):
            any_different = True
            break

    assert any_different, "shuffled and as_stored ordering produced identical samples for all items"


def test_no_token_gaps_or_overlaps(simple_env):
    """Consecutive samples should tile the token stream without gaps.

    We verify this by checking that sample[i+1][0] == sample[i][-1],
    i.e. the last token of sample i equals the first token of sample i+1
    (they share the boundary token since we use seq_len+1 per sample).
    """
    seq_len = 5
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    num_samples = (simple_env["total_tokens"] - 1) // seq_len  # 16
    ds = _build_dataset(config, Split.train, num_samples)

    prev = None
    for i in range(len(ds)):
        curr = ds[i]["text"]
        if prev is not None:
            assert prev[-1].item() == curr[0].item(), (
                f"gap/overlap at boundary between sample {i-1} and {i}: "
                f"prev[-1]={prev[-1].item()}, curr[0]={curr[0].item()}"
            )
        prev = curr


def _reference_samples(ds: StitchedSequenceDataset) -> dict[int, torch.Tensor]:
    """Collect all samples from the dataset in the main process (num_workers=0)."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return {i: batch["text"][0] for i, batch in enumerate(loader)}


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_multiworker_matches_single_worker(simple_env, num_workers):
    """DataLoader with num_workers>0 must produce the same tokens as num_workers=0.

    This exercises:
    - StitchedSequenceDataset.__getstate__/__setstate__ (dataset pickling)
    - ShardStore re-initialisation in each worker process
    - No race conditions when multiple workers open the same shard files
    """
    seq_len = 5
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    num_samples = (simple_env["total_tokens"] - 1) // seq_len  # 16

    # Build the dataset once; reuse the same sample_index for both loaders
    stitched_seq, sample_index = build_sample_index(config, Split.train, num_samples, caching_allowed=False)
    ds_ref = StitchedSequenceDataset(config, Split.train, stitched_seq, sample_index)
    ds_mw = StitchedSequenceDataset(config, Split.train, stitched_seq, sample_index)

    reference = _reference_samples(ds_ref)

    loader_mw = DataLoader(
        ds_mw,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn",
    )
    for i, batch in enumerate(loader_mw):
        tokens = batch["text"][0]
        assert torch.equal(tokens, reference[i]), (
            f"worker mismatch at sample {i} with num_workers={num_workers}: "
            f"expected {reference[i].tolist()}, got {tokens.tolist()}"
        )


def test_multiworker_no_missing_or_duplicate_samples(simple_env):
    """All sample indices must be returned exactly once by the multi-worker loader."""
    seq_len = 5
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    num_samples = (simple_env["total_tokens"] - 1) // seq_len

    ds = _build_dataset(config, Split.train, num_samples)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        multiprocessing_context="spawn",
    )
    collected = [batch["text"][0] for batch in loader]

    assert len(collected) == num_samples, f"expected {num_samples} samples, got {len(collected)}"


def test_multiworker_shard_store_is_independent_per_worker(simple_env):
    """ShardStore LRU cache should not be shared between workers.

    We verify this indirectly: with many workers all reading the same shards,
    there should be no exceptions and results should be correct.
    Exercises the re-init path when two workers open the same .idx/.bin files.
    """
    seq_len = 3
    config = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    num_samples = (simple_env["total_tokens"] - 1) // seq_len

    ds = _build_dataset(config, Split.train, num_samples)

    # Use more workers than collections to ensure multiple workers access the same shards
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        multiprocessing_context="spawn",
    )

    total = 0
    for batch in loader:
        assert batch["text"].shape[-1] == seq_len + 1
        assert batch["text"].dtype == torch.int64
        total += batch["text"].shape[0]

    assert total == num_samples


def test_shuffled_ordering_preserves_token_multiset(simple_env):
    """Shuffled ordering reorders doc chunks but must contain the same tokens as as_stored.

    This verifies the bug fix: sample_index boundaries are computed in as_stored order,
    so docs are always read in as_stored order first (preserving correct t0/t1 offsets
    for boundary docs), then chunks are shuffled. The multiset of tokens per sample
    must therefore be identical between as_stored and shuffled.
    """
    seq_len = 12
    config_base = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config_base, Split.train, 5, caching_allowed=False)

    config_as_stored = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    config_shuffled = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.shuffled,
        split_ratio=(1.0, 0.0, 0.0),
    )

    ds_as_stored = StitchedSequenceDataset(config_as_stored, Split.train, stitched_seq, sample_index)
    ds_shuffled = StitchedSequenceDataset(config_shuffled, Split.train, stitched_seq, sample_index)

    for i in range(len(ds_as_stored)):
        tokens_as_stored = sorted(ds_as_stored[i]["text"].tolist())
        tokens_shuffled = sorted(ds_shuffled[i]["text"].tolist())
        assert tokens_as_stored == tokens_shuffled, (
            f"sample {i}: shuffled ordering changed the token multiset. "
            f"as_stored={tokens_as_stored}, shuffled={tokens_shuffled}. "
            "This indicates boundary docs (t0/t1 offsets) are being applied to wrong docs."
        )


def test_reversed_ordering_reverses_chunks(simple_env):
    """reversed ordering should produce chunks in reverse order vs as_stored."""
    seq_len = 15
    config_base = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config_base, Split.train, 4, caching_allowed=False)

    config_as_stored = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    config_reversed = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.reversed,
        split_ratio=(1.0, 0.0, 0.0),
    )

    ds_as_stored = StitchedSequenceDataset(config_as_stored, Split.train, stitched_seq, sample_index)
    ds_reversed = StitchedSequenceDataset(config_reversed, Split.train, stitched_seq, sample_index)

    # reversed must contain the same token multiset as as_stored (same docs, reversed chunk order)
    for i in range(len(ds_as_stored)):
        tokens_as_stored = sorted(ds_as_stored[i]["text"].tolist())
        tokens_reversed = sorted(ds_reversed[i]["text"].tolist())
        assert tokens_as_stored == tokens_reversed, f"sample {i}: reversed ordering changed the token multiset."

    # and at least some samples should have different order
    any_different = any(
        not torch.equal(ds_as_stored[i]["text"], ds_reversed[i]["text"]) for i in range(len(ds_as_stored))
    )
    assert any_different, "reversed and as_stored produced identical token order for all samples"


def test_reversed_ordering_reverses_chunks_directly(simple_env):
    """Reversed ordering swaps doc chunks within a collection.

    col 0 has 3 docs of 5 tokens each. seq_len=9 gives one 10-token sample
    spanning exactly doc 0 (5 tokens) and doc 1 (5 tokens) — both full chunks.
    reversed should produce [doc1_tokens, doc0_tokens].
    """
    # seq_len=9 → (15-1)//9 = 1 sample spanning doc 0 and doc 1 of col 0
    seq_len = 9
    config_base = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        split_ratio=(0.25, 0.75, 0.0),  # train = col 0 only (1 of 4 collections)
    )
    stitched_seq, sample_index = build_sample_index(config_base, Split.train, 1, caching_allowed=False)

    config_as_stored = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(0.25, 0.75, 0.0),
    )
    config_reversed = StitchedDatasetConfig(
        stitched_seq_path=simple_env["stitched_seq_path"],
        tokenized_data_root=simple_env["data_root"],
        sequence_length=seq_len,
        ordering_strategy=OrderingStrategy.reversed,
        split_ratio=(0.25, 0.75, 0.0),
    )

    ds_as_stored = StitchedSequenceDataset(config_as_stored, Split.train, stitched_seq, sample_index)
    ds_reversed = StitchedSequenceDataset(config_reversed, Split.train, stitched_seq, sample_index)

    tokens_as_stored = ds_as_stored[0]["text"].tolist()
    tokens_reversed = ds_reversed[0]["text"].tolist()

    # Both docs are 5 tokens; reversed swaps the two chunks
    assert len(tokens_as_stored) == seq_len + 1 == 10
    assert tokens_reversed[:5] == tokens_as_stored[5:], (
        f"reversed first chunk should be as_stored second chunk\n"
        f"  as_stored: {tokens_as_stored}\n"
        f"  reversed:  {tokens_reversed}"
    )
    assert tokens_reversed[5:] == tokens_as_stored[:5], (
        f"reversed second chunk should be as_stored first chunk\n"
        f"  as_stored: {tokens_as_stored}\n"
        f"  reversed:  {tokens_reversed}"
    )
