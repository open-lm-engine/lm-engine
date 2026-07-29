# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Unit tests for MixedSequenceDataset and the mixed dataset builder.

Self-contained: fake in-memory sources are generic children used to exercise mixture
mechanics (a numpy-returning child with an extra key exercises dtype normalization
and key whitelisting), and a tiny stitched corpus built via the stitched test helpers
provides a real child.

No GPUs, no distributed state, no C++ compilation required.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lm_engine.data.megatron import Split
from lm_engine.data.megatron.sampler import MegatronBatchSampler
from lm_engine.data.mixed import MixedSequenceDataset, build_mixed_datasets
from lm_engine.data.mixed.builder import _validate_sources
from lm_engine.data.stitched.builder import build_sample_index
from lm_engine.data.stitched.config import OrderingStrategy, StitchedDatasetConfig
from lm_engine.data.stitched.dataset import StitchedSequenceDataset

from .stitched_dataset_test import _make_shard, _make_stitched_seq


SEQ_LEN = 5


class _FakeNumpySource(torch.utils.data.Dataset):
    """A child returning numpy int64 samples plus an extra key, exercising
    MixedSequenceDataset's dtype normalization and extra-key dropping.

    Every token of sample i is `base + i`, so a sample's first token identifies the
    (source, local index) pair exactly. Like StitchedSequenceDataset, __getitem__ accepts
    idx >= num_samples and wraps around -- MixedSequenceDataset's single-source path passes
    the raw global idx straight through to the child, relying on the child itself to handle
    epoch replay.
    """

    def __init__(self, num_samples: int, base: int) -> None:
        self._num_samples = num_samples
        self._base = base

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict:
        idx = idx % self._num_samples
        return {
            "dataset_id": 0,
            "text": np.full(SEQ_LEN + 1, self._base + idx, dtype=np.int64),
        }


class _FakeTensorSource(_FakeNumpySource):
    """Stands in for StitchedSequenceDataset: torch tensor samples, no extra keys."""

    def __getitem__(self, idx: int) -> dict:
        return {"text": torch.as_tensor(super().__getitem__(idx)["text"])}


def _make_mixture(num_samples: int, seed: int = 42) -> MixedSequenceDataset:
    """30-sample numpy source (tokens 0..29) + 70-sample tensor source (tokens 1000..1069)."""
    return MixedSequenceDataset(
        datasets=[_FakeNumpySource(30, base=0), _FakeTensorSource(70, base=1000)],
        data_names=["a", "b"],
        num_samples=num_samples,
        seed=seed,
    )


def _first_tokens(ds: MixedSequenceDataset, start: int, stop: int) -> list[int]:
    return [int(ds[i]["text"][0]) for i in range(start, stop)]


def _make_stitched_source(
    tmp_path,
    name: str,
    num_docs: int,
    token_base: int,
    split_ratio: tuple = (1.0, 0.0, 0.0),
) -> dict:
    """Build a tiny stitched corpus (num_docs docs of 5 tokens, one collection per 2 docs)
    and return the class_args dict for a mixed stitched source.
    """
    data_root = str(tmp_path / f"{name}_data")
    docs = [[token_base + d * 10 + i for i in range(5)] for d in range(num_docs)]
    _make_shard(f"{data_root}/{name}", f"{name}_content", docs)

    stitched_seq_path = str(tmp_path / f"{name}_stitched_seq.parquet")
    _make_stitched_seq(
        stitched_seq_path,
        [
            {
                "shard": f"{name}/{name}_content",
                "coll_beg": beg,
                "coll_end": min(beg + 2, num_docs),
            }
            for beg in range(0, num_docs, 2)
        ],
    )

    return {
        "stitched_seq_path": stitched_seq_path,
        "tokenized_data_root": data_root,
        "split_ratio": list(split_ratio),
        "caching_allowed": False,
    }


def test_epoch_coverage_and_replay():
    """Each mixture epoch visits every sample of every source exactly once; replayed
    epochs cover the same samples in a different order (when reshuffle_per_epoch=True).
    """
    ds = _make_mixture(num_samples=200)
    assert len(ds) == 200

    expected = set(range(30)) | set(range(1000, 1070))

    first_epoch = _first_tokens(ds, 0, 100)
    assert len(first_epoch) == len(set(first_epoch)) == 100
    assert set(first_epoch) == expected

    second_epoch = _first_tokens(ds, 100, 200)
    assert set(second_epoch) == expected
    assert first_epoch != second_epoch


def test_no_reshuffle_replay():
    """When reshuffle_per_epoch=False (for val/test), replayed epochs maintain the
    exact same deterministic order.
    """
    ds = MixedSequenceDataset(
        datasets=[_FakeNumpySource(30, base=0), _FakeTensorSource(70, base=1000)],
        data_names=["a", "b"],
        num_samples=200,  # 2 epochs
        seed=42,
        reshuffle_per_epoch=False,
    )
    assert len(ds) == 200

    # Get first epoch samples
    first_epoch = _first_tokens(ds, 0, 100)

    # Get second epoch samples - should be IDENTICAL order, not reshuffled
    second_epoch = _first_tokens(ds, 100, 200)

    # Verify exact same order (not just same set of tokens)
    assert first_epoch == second_epoch


def test_mixing_is_size_proportional():
    """Sources should be spread through the epoch roughly proportionally to their size."""
    ds = _make_mixture(num_samples=100)
    first_half = _first_tokens(ds, 0, 50)
    count_a = sum(token < 30 for token in first_half)
    # Source a is 30% of the mixture; deterministic given the seed, generous bounds
    # to document intent rather than the exact permutation.
    assert 5 <= count_a <= 25


def test_determinism_and_seed():
    ds_1 = _make_mixture(num_samples=100)
    ds_2 = _make_mixture(num_samples=100)
    assert _first_tokens(ds_1, 0, 100) == _first_tokens(ds_2, 0, 100)

    ds_3 = _make_mixture(num_samples=100, seed=7)
    assert _first_tokens(ds_1, 0, 100) != _first_tokens(ds_3, 0, 100)


def test_single_source_mixture_preserves_order():
    """With exactly one source, there is nothing to mix: MixedSequenceDataset must not apply
    its own shuffle on top -- samples are served in the source's own order, every replayed
    epoch, regardless of seed.
    """

    def _make(seed: int) -> MixedSequenceDataset:
        return MixedSequenceDataset(
            datasets=[_FakeNumpySource(50, base=0)],
            data_names=["only"],
            num_samples=150,  # 3 replays of a 50-sample epoch
            seed=seed,
        )

    expected = list(range(50)) * 3
    assert _first_tokens(_make(42), 0, 150) == expected
    assert _first_tokens(_make(7), 0, 150) == expected  # seed-independent


@pytest.mark.parametrize("consumed_samples", [52, 100])
def test_resume_equivalence(consumed_samples):
    """Rebuilding the dataset and resuming from consumed_samples must reproduce the
    tail of a from-scratch run (including across the epoch boundary at 100).
    """
    micro_batch_size = 4

    def _batches(consumed: int) -> list[torch.Tensor]:
        ds = _make_mixture(num_samples=200)
        loader = DataLoader(
            ds,
            batch_sampler=MegatronBatchSampler(
                total_samples=len(ds),
                consumed_samples=consumed,
                micro_batch_size=micro_batch_size,
                num_replicas=1,
                rank=0,
            ),
        )
        return [batch["text"] for batch in loader]

    reference = _batches(0)
    resumed = _batches(consumed_samples)

    assert len(resumed) == len(reference) - consumed_samples // micro_batch_size
    for reference_batch, resumed_batch in zip(reference[consumed_samples // micro_batch_size :], resumed):
        assert torch.equal(reference_batch, resumed_batch)


def test_mixed_dtype_collate_and_key_whitelist(tmp_path):
    """Batches spanning a numpy-returning source and a real stitched (tensor-returning)
    source must collate into uniform int64 tensors with only the "text" key.
    """
    stitched_args = _make_stitched_source(tmp_path, "corpus", num_docs=16, token_base=100000)
    config = StitchedDatasetConfig(
        stitched_seq_path=stitched_args["stitched_seq_path"],
        tokenized_data_root=stitched_args["tokenized_data_root"],
        sequence_length=SEQ_LEN,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config, Split.train, None, caching_allowed=False)
    stitched_ds = StitchedSequenceDataset(config, Split.train, stitched_seq, sample_index)

    ds = MixedSequenceDataset(
        datasets=[_FakeNumpySource(25, base=0), stitched_ds],
        data_names=["fake", "stitched"],
        num_samples=25 + len(stitched_ds),
        seed=42,
    )

    seen_sources = set()
    for batch in DataLoader(ds, batch_size=8, shuffle=False):
        assert set(batch.keys()) == {"text"}
        assert batch["text"].dtype == torch.int64
        assert batch["text"].shape[-1] == SEQ_LEN + 1
        seen_sources.update("stitched" if token >= 100000 else "fake" for token in batch["text"][:, 0].tolist())

    assert seen_sources == {"fake", "stitched"}


@pytest.mark.parametrize("num_workers", [2, 4])
def test_multiworker_matches_single_worker(tmp_path, num_workers):
    """Multi-worker loading must match single-process loading (exercises the
    permutation-cache __getstate__ path and child re-initialisation under spawn).
    """
    stitched_args = _make_stitched_source(tmp_path, "corpus", num_docs=16, token_base=100000)
    config = StitchedDatasetConfig(
        stitched_seq_path=stitched_args["stitched_seq_path"],
        tokenized_data_root=stitched_args["tokenized_data_root"],
        sequence_length=SEQ_LEN,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config, Split.train, None, caching_allowed=False)

    def _make_ds() -> MixedSequenceDataset:
        stitched_ds = StitchedSequenceDataset(config, Split.train, stitched_seq, sample_index)
        return MixedSequenceDataset(
            datasets=[_FakeNumpySource(25, base=0), stitched_ds],
            data_names=["fake", "stitched"],
            num_samples=2 * (25 + len(stitched_ds)),
            seed=42,
        )

    reference = [batch["text"] for batch in DataLoader(_make_ds(), batch_size=4, shuffle=False, num_workers=0)]

    loader = DataLoader(
        _make_ds(),
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn",
    )
    for i, batch in enumerate(loader):
        assert torch.equal(batch["text"], reference[i]), f"worker mismatch at batch {i}"
    assert i == len(reference) - 1


def test_single_source_mixture_matches_child_order(tmp_path):
    """A one-source MixedDataset defers entirely to the child's own order: indexing the
    mixture must return exactly the same samples as indexing the child directly, including
    across replayed epochs.
    """
    stitched_args = _make_stitched_source(tmp_path, "solo", num_docs=16, token_base=100000)
    config = StitchedDatasetConfig(
        stitched_seq_path=stitched_args["stitched_seq_path"],
        tokenized_data_root=stitched_args["tokenized_data_root"],
        sequence_length=SEQ_LEN,
        ordering_strategy=OrderingStrategy.as_stored,
        split_ratio=(1.0, 0.0, 0.0),
    )
    stitched_seq, sample_index = build_sample_index(config, Split.train, None, caching_allowed=False)
    stitched_ds = StitchedSequenceDataset(config, Split.train, stitched_seq, sample_index)

    mixed_ds = MixedSequenceDataset(
        datasets=[stitched_ds],
        data_names=["solo"],
        num_samples=3 * len(stitched_ds),
        seed=42,
    )

    for i in range(len(mixed_ds)):
        assert torch.equal(mixed_ds[i]["text"], stitched_ds[i]["text"])


def test_mixed_dataloader_two_stitched_sources_expected_batches(tmp_path):
    """Two hand-built stitched sources, combined through MixedSequenceDataset + DataLoader.

    Dataset 1: 2 examples, each its own collection. Example 1 = tokens 1..10, example 2 =
    tokens 11..30. stitch order (stitched_seq row order) lists example 2's collection first,
    example 1's second -- so the flat token stream is [11..30, 1..10] (30 tokens). With
    sequence_length=10 this tiles into exactly 2 samples: [11..21] and [21..30, 1].

    Dataset 2: 4 examples, docs 1-3 grouped into one collection, doc 4 its own collection, in
    forward stitch order -- so the flat token stream is simply [1..40] (40 tokens), tiling
    into exactly 3 samples: [1..11], [11..21], [21..31].

    StitchedSequenceDataset.__getitem__ applies its own per-epoch RNG permutation on top of
    stream order, so the above stream-order values are checked via _fetch_tokens (which
    bypasses that permutation, same technique as test_no_token_gaps_or_overlaps in
    stitched_dataset_test.py) -- not via ds[i] directly.
    """
    seq_len = 10

    # ---- Dataset 1: 2 examples, each its own collection, stitch order reversed ----
    ds1_root = str(tmp_path / "ds1_data")
    example_1 = list(range(1, 11))  # tokens 1..10
    example_2 = list(range(11, 31))  # tokens 11..30
    _make_shard(f"{ds1_root}/ds1", "ds1_content", [example_1, example_2])  # doc0=ex1, doc1=ex2

    ds1_seq_path = str(tmp_path / "ds1_stitched_seq.parquet")
    _make_stitched_seq(
        ds1_seq_path,
        [
            {
                "shard": "ds1/ds1_content",
                "coll_beg": 1,
                "coll_end": 2,
            },  # example 2's collection, first
            {
                "shard": "ds1/ds1_content",
                "coll_beg": 0,
                "coll_end": 1,
            },  # example 1's collection, second
        ],
    )
    ds1_config = StitchedDatasetConfig(
        stitched_seq_path=ds1_seq_path,
        tokenized_data_root=ds1_root,
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    ds1_seq, ds1_index = build_sample_index(ds1_config, Split.train, None, caching_allowed=False)
    ds1 = StitchedSequenceDataset(ds1_config, Split.train, ds1_seq, ds1_index)

    assert len(ds1) == 2
    assert ds1._fetch_tokens(0).tolist() == list(range(11, 22))
    assert ds1._fetch_tokens(1).tolist() == list(range(21, 31)) + [1]

    # ---- Dataset 2: 4 examples, docs 1-3 one collection, doc 4 its own, forward stitch order ----
    ds2_root = str(tmp_path / "ds2_data")
    ex1, ex2, ex3, ex4 = (
        list(range(1, 11)),
        list(range(11, 21)),
        list(range(21, 31)),
        list(range(31, 41)),
    )
    _make_shard(f"{ds2_root}/ds2", "ds2_content", [ex1, ex2, ex3, ex4])

    ds2_seq_path = str(tmp_path / "ds2_stitched_seq.parquet")
    _make_stitched_seq(
        ds2_seq_path,
        [
            {
                "shard": "ds2/ds2_content",
                "coll_beg": 0,
                "coll_end": 3,
            },  # examples 1-3's collection
            {
                "shard": "ds2/ds2_content",
                "coll_beg": 3,
                "coll_end": 4,
            },  # example 4's collection
        ],
    )
    ds2_config = StitchedDatasetConfig(
        stitched_seq_path=ds2_seq_path,
        tokenized_data_root=ds2_root,
        sequence_length=seq_len,
        split_ratio=(1.0, 0.0, 0.0),
    )
    ds2_seq, ds2_index = build_sample_index(ds2_config, Split.train, None, caching_allowed=False)
    ds2 = StitchedSequenceDataset(ds2_config, Split.train, ds2_seq, ds2_index)

    assert len(ds2) == 3
    assert ds2._fetch_tokens(0).tolist() == list(range(1, 12))
    assert ds2._fetch_tokens(1).tolist() == list(range(11, 22))
    assert ds2._fetch_tokens(2).tolist() == list(range(21, 32))

    # ---- Mixed dataloader over both stitched sources ----
    num_samples = len(ds1) + len(ds2)  # exactly one mixture epoch, no replay
    mixed_ds = MixedSequenceDataset(datasets=[ds1, ds2], data_names=["ds1", "ds2"], num_samples=num_samples, seed=42)

    # Reconstruct the expected order using the mixture's own epoch-0 permutation and offsets
    # (mirrors MixedSequenceDataset.__getitem__ exactly), resolving each sample via a live
    # ds1[i]/ds2[i] call rather than hand-derived literals.
    offsets = [0, len(ds1), len(ds1) + len(ds2)]
    per_child_samples = [
        [ds1[i]["text"] for i in range(len(ds1))],
        [ds2[i]["text"] for i in range(len(ds2))],
    ]
    permutation = mixed_ds._get_permutation(0)
    expected_samples = []
    for position in permutation:
        dataset_id = int(np.searchsorted(offsets, position, side="right")) - 1
        expected_samples.append(per_child_samples[dataset_id][position - offsets[dataset_id]])

    batch_size = 2
    expected_batches = [torch.stack(expected_samples[i : i + batch_size]) for i in range(0, num_samples, batch_size)]
    actual_batches = [batch["text"] for batch in DataLoader(mixed_ds, batch_size=batch_size, shuffle=False)]

    assert len(actual_batches) == len(expected_batches) == 3  # batch sizes [2, 2, 1]
    for actual, expected in zip(actual_batches, expected_batches):
        assert torch.equal(actual, expected)


def _valid_class_args() -> dict:
    return {
        "sequence_length": SEQ_LEN,
        "sources": [
            {
                "class_name": "StitchedDataset",
                "data_name": "a",
                "class_args": {"stitched_seq_path": "x"},
            },
            {
                "class_name": "StitchedDataset",
                "data_name": "b",
                "class_args": {"stitched_seq_path": "y"},
            },
        ],
    }


def test_validate_sources_accepts_valid_config():
    _validate_sources(_valid_class_args())


def test_validate_sources_rejections():
    class_args = _valid_class_args()
    class_args["sources"] = []
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    class_args = _valid_class_args()
    del class_args["sequence_length"]
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    class_args = _valid_class_args()
    class_args["sources"][0]["class_name"] = "HuggingFaceDataset"
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    # mixing is size-proportional, sampling ratios are not supported
    class_args = _valid_class_args()
    class_args["sources"][0]["data_sampling_ratio"] = 7
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    # sequence_length is global only
    class_args = _valid_class_args()
    class_args["sources"][0]["class_args"]["sequence_length"] = 2 * SEQ_LEN
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    class_args = _valid_class_args()
    class_args["sources"][1]["data_name"] = "a"
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    # megatron sources are no longer supported -- stitched only
    class_args = _valid_class_args()
    class_args["sources"][1]["class_name"] = "MegatronDataset"
    class_args["sources"][1]["class_args"] = {"data_path": ["/prefix"]}
    with pytest.raises(AssertionError):
        _validate_sources(class_args)

    # duplicate stitched_seq_path across sources is rejected -- no two sources may mix in the same corpus
    class_args = _valid_class_args()
    class_args["sources"][1]["class_args"]["stitched_seq_path"] = class_args["sources"][0]["class_args"][
        "stitched_seq_path"
    ]
    with pytest.raises(AssertionError):
        _validate_sources(class_args)


def test_build_mixed_datasets(tmp_path):
    """End-to-end builder test with two stitched sources: one-epoch children,
    size-proportional train mixture, and a single mixed val dataset combining samples
    from sources with non-zero val splits (source "small" has a val split, source
    "large" does not).
    """
    val_samples = 8
    class_args = {
        "sequence_length": SEQ_LEN,
        "sources": [
            {
                "class_name": "StitchedDataset",
                "data_name": "small",
                "class_args": _make_stitched_source(
                    tmp_path,
                    "small",
                    num_docs=8,
                    token_base=0,
                    split_ratio=(0.75, 0.25, 0.0),
                ),
            },
            {
                "class_name": "StitchedDataset",
                "data_name": "large",
                "class_args": _make_stitched_source(tmp_path, "large", num_docs=24, token_base=100000),
            },
        ],
    }

    train_samples = 64
    train_ds, val_ds, test_ds = build_mixed_datasets(
        class_args,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=8,
        default_seed=42,
    )

    # small: 4 collections of 10 tokens, 3 train + 1 val -> (30 - 1) // 5 = 5 train samples/epoch;
    # large: 24 docs, all train -> (120 - 1) // 5 = 23
    assert isinstance(train_ds, MixedSequenceDataset)
    assert len(train_ds) == train_samples
    assert train_ds._epoch_samples == 5 + 23
    assert train_ds._reshuffle_per_epoch is True  # Training should reshuffle

    # Validation: single MixedSequenceDataset combining all sources with non-zero val splits
    # Only "small" has validation, so val_ds should be a MixedSequenceDataset with one child
    assert isinstance(val_ds, MixedSequenceDataset)
    assert len(val_ds) == val_samples
    assert val_ds._reshuffle_per_epoch is False  # Val/test should NOT reshuffle

    # Test: no sources have test splits
    assert test_ds is None

    val_sample = val_ds[0]["text"]
    assert val_sample.dtype == torch.int64
    assert val_sample.shape == (SEQ_LEN + 1,)

    train_sample = train_ds[0]["text"]
    assert train_sample.dtype == torch.int64
    assert train_sample.shape == (SEQ_LEN + 1,)

    # both sources present in the train first epoch, in their size proportions
    first_epoch = {int(train_ds[i]["text"][0]) for i in range(train_ds._epoch_samples)}
    assert any(token < 100000 for token in first_epoch)
    assert any(token >= 100000 for token in first_epoch)
