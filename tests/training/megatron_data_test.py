# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
import tempfile

import numpy as np
import torch

from lm_engine.data.megatron.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

from .test_commons import TestCommons


class MegatronDatasetTest(TestCommons):
    def test_megatron_dataset_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "file")
            bin_path = get_bin_path(prefix)
            idx_path = get_idx_path(prefix)

            num_documents = 1000
            document = np.array([1, 2])
            self._build_dataset(bin_path, idx_path, num_documents, document)

            assert os.path.exists(bin_path)
            assert os.path.exists(idx_path)

            dataset = MMapIndexedDataset(prefix)

            assert len(dataset) == num_documents
            for i in dataset:
                assert (i == document).all()

    def test_megatron_dataset_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix1 = os.path.join(tmpdir, "file1")
            bin_path1 = get_bin_path(prefix1)
            idx_path1 = get_idx_path(prefix1)

            num_documents1 = 1000
            document1 = np.array([1, 2])
            self._build_dataset(bin_path1, idx_path1, num_documents1, document1)

            prefix2 = os.path.join(tmpdir, "file2")
            bin_path2 = get_bin_path(prefix2)
            idx_path2 = get_idx_path(prefix2)

            num_documents2 = 2000
            document2 = np.array([3, 4, 5])
            self._build_dataset(bin_path2, idx_path2, num_documents2, document2)

            prefix_merged = os.path.join(tmpdir, "merged")
            bin_path_merged = get_bin_path(prefix_merged)
            idx_path_merged = get_idx_path(prefix_merged)

            builder = MMapIndexedDatasetBuilder(bin_path_merged)
            builder.add_index(prefix1)
            builder.add_index(prefix2)
            builder.finalize(idx_path_merged)

            dataset = MMapIndexedDataset(prefix_merged)

            assert len(dataset) == num_documents1 + num_documents2
            for i, d in enumerate(dataset):
                assert (d == (document1 if i < num_documents1 else document2)).all()

    def _build_dataset(self, bin_path: str, idx_path: str, num_documents: int, repeated_document: np.ndarray) -> None:
        builder = MMapIndexedDatasetBuilder(bin_path)
        repeated_document = torch.tensor(repeated_document)

        for _ in range(num_documents):
            builder.add_item(repeated_document)
            builder.end_document()

        builder.finalize(idx_path)
