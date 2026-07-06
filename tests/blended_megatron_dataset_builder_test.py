# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
import tempfile

import numpy as np
import pytest
import torch

from lm_engine.data.megatron.blended_dataset import BlendedDataset, ConcatenatedDataset
from lm_engine.data.megatron.blended_megatron_dataset_builder import (
    _get_blend_from_list,
    _get_num_samples_per_prefix,
    _get_sizes_for_blend,
    _resolve_idx_path,
    build,
)
from lm_engine.data.megatron.blended_megatron_dataset_config import GPTDatasetConfig
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDatasetBuilder, get_bin_path, get_idx_path
from lm_engine.data.megatron.utils import compile_helpers, normalize
from lm_engine.parallel import ProcessGroupManager

from .utils import slow_test


def _force_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # build()/compile_helpers() are only ever exercised in this repo under torchrun, where
    # ProcessGroupManager is fully initialized (mesh, ranks, ...). Standing that up for a
    # single-process unit test would need a real distributed bootstrap for no benefit here, so
    # instead force the same non-distributed code paths a rank-0-only, world-size-1 run would take.
    monkeypatch.setattr(ProcessGroupManager, "is_initialized", staticmethod(lambda: False))
    monkeypatch.setattr(ProcessGroupManager, "get_global_rank", staticmethod(lambda: 0))
    monkeypatch.setattr(ProcessGroupManager, "barrier", staticmethod(lambda: None))


@pytest.mark.parametrize(
    "blend, expected_prefixes, expected_weights",
    [
        (["path1"], ["path1"], None),
        (["path1", "path2"], ["path1", "path2"], None),
        (["path1", "path2", "path3"], ["path1", "path2", "path3"], None),
        (["  path1  ", "  path2  "], ["path1", "path2"], None),
        (["30", "path1", "70", "path2"], ["path1", "path2"], [30.0, 70.0]),
        (["30", " path1 ", "70", " path2 "], ["path1", "path2"], [30.0, 70.0]),
    ],
)
def test_get_blend_from_list(
    blend: list[str], expected_prefixes: list[str], expected_weights: list[float] | None
) -> None:
    prefixes, weights = _get_blend_from_list(blend)

    assert prefixes == expected_prefixes
    assert weights == expected_weights


def test_get_sizes_for_blend() -> None:
    weights = [0.5, 0.5]
    target_num_samples_per_split = [100, 10, 0]

    sizes_per_dataset = _get_sizes_for_blend(weights, target_num_samples_per_split)

    assert sizes_per_dataset == [
        [int(np.ceil(100 * 0.5 * 1.005)), int(np.ceil(10 * 0.5 * 1.005)), 0],
        [int(np.ceil(100 * 0.5 * 1.005)), int(np.ceil(10 * 0.5 * 1.005)), 0],
    ]


def test_resolve_idx_path_returns_none_for_local_prefix() -> None:
    assert _resolve_idx_path("/some/local/path", node_uses_local_storage=False, config=None) is None


def _build_dataset(bin_path: str, idx_path: str, num_documents: int, document: np.ndarray) -> None:
    builder = MMapIndexedDatasetBuilder(bin_path)
    document = torch.tensor(document)

    for _ in range(num_documents):
        builder.add_item(document)
        builder.end_document()

    builder.finalize(idx_path)


def test_get_num_samples_per_prefix() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix1 = os.path.join(tmpdir, "dataset1")
        prefix2 = os.path.join(tmpdir, "dataset2")

        _build_dataset(get_bin_path(prefix1), get_idx_path(prefix1), num_documents=200, document=np.arange(8))
        _build_dataset(get_bin_path(prefix2), get_idx_path(prefix2), num_documents=400, document=np.arange(8))

        num_samples = _get_num_samples_per_prefix([prefix1, prefix2], node_uses_local_storage=False, config=None)

        assert num_samples == [200, 400]


def _expected_sizes(num_samples_per_prefix: list[int], target_num_samples: int) -> list[int]:
    weights = normalize(num_samples_per_prefix)
    return [sizes[0] for sizes in _get_sizes_for_blend(weights, [target_num_samples])]


@slow_test
def test_build_with_unweighted_blend_is_concatenated_by_num_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_single_process(monkeypatch)
    compile_helpers()

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix1 = os.path.join(tmpdir, "dataset1")
        prefix2 = os.path.join(tmpdir, "dataset2")

        # unequal dataset sizes: an unweighted blend should be concatenated as-is, contributing
        # samples proportional to 200:400, not split equally
        _build_dataset(get_bin_path(prefix1), get_idx_path(prefix1), num_documents=200, document=np.arange(8))
        _build_dataset(get_bin_path(prefix2), get_idx_path(prefix2), num_documents=400, document=np.arange(8))

        config = GPTDatasetConfig(
            sequence_length=4,
            blend=[prefix1, prefix2],
            split="100,0,0",
            path_to_cache=os.path.join(tmpdir, "cache"),
        )

        train_dataset, val_dataset, test_dataset = build(
            sizes=[50, 0, 0],
            config=config,
            tokenizer=None,
            node_uses_local_storage=False,
            random_seed=1234,
        )

        assert val_dataset is None
        assert test_dataset is None

        assert isinstance(train_dataset, ConcatenatedDataset)

        expected_sizes = _expected_sizes([200, 400], 50)
        assert [len(dataset) for dataset in train_dataset.datasets] == expected_sizes
        assert len(train_dataset) == sum(expected_sizes)


@slow_test
def test_build_with_unweighted_blend_per_split_is_concatenated_by_num_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_single_process(monkeypatch)
    compile_helpers()

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix1 = os.path.join(tmpdir, "dataset1")
        prefix2 = os.path.join(tmpdir, "dataset2")

        _build_dataset(get_bin_path(prefix1), get_idx_path(prefix1), num_documents=200, document=np.arange(8))
        _build_dataset(get_bin_path(prefix2), get_idx_path(prefix2), num_documents=400, document=np.arange(8))

        config = GPTDatasetConfig(
            sequence_length=4,
            blend_per_split=[[prefix1, prefix2], None, None],
            path_to_cache=os.path.join(tmpdir, "cache"),
        )

        train_dataset, val_dataset, test_dataset = build(
            sizes=[50, 0, 0],
            config=config,
            tokenizer=None,
            node_uses_local_storage=False,
            random_seed=1234,
        )

        assert val_dataset is None
        assert test_dataset is None

        assert isinstance(train_dataset, ConcatenatedDataset)

        expected_sizes = _expected_sizes([200, 400], 50)
        assert [len(dataset) for dataset in train_dataset.datasets] == expected_sizes
        assert len(train_dataset) == sum(expected_sizes)


@slow_test
def test_build_with_explicitly_weighted_blend_uses_blended_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_single_process(monkeypatch)
    compile_helpers()

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix1 = os.path.join(tmpdir, "dataset1")
        prefix2 = os.path.join(tmpdir, "dataset2")

        _build_dataset(get_bin_path(prefix1), get_idx_path(prefix1), num_documents=200, document=np.arange(8))
        _build_dataset(get_bin_path(prefix2), get_idx_path(prefix2), num_documents=400, document=np.arange(8))

        config = GPTDatasetConfig(
            sequence_length=4,
            # explicit weights ignore the datasets' actual sizes: this should still be a
            # weighted random blend, not a concatenation
            blend=["30", prefix1, "70", prefix2],
            split="100,0,0",
            path_to_cache=os.path.join(tmpdir, "cache"),
        )

        train_dataset, val_dataset, test_dataset = build(
            sizes=[50, 0, 0],
            config=config,
            tokenizer=None,
            node_uses_local_storage=False,
            random_seed=1234,
        )

        assert val_dataset is None
        assert test_dataset is None

        assert isinstance(train_dataset, BlendedDataset)
        assert train_dataset.weights == normalize([30.0, 70.0])
