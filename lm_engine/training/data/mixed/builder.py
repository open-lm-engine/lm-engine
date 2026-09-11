# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Builders for the MixedDataset pretraining data pipeline.

A MixedDataset draws from multiple sources, each a StitchedDataset, sharing one global
sequence_length. One mixture epoch visits every sample of every source exactly once.

Validation and test are also mixed -- a single MixedSequenceDataset combines val (or test) samples
from all sources that provide the split. Each mixed val/test dataset uses the same deterministic
shuffle as training (seeded from the config) but replays in the same order (no per-epoch reshuffle).
The trainer reports aggregate metrics for the mixed val/test splits.

Expected config layout (a single entry in the training config's datasets list):

    datasets:
    - class_name: MixedDataset
      data_name: Mix
      class_args:
        sequence_length: 65536   # global, applies to all sources
        eval_steps: 2
        num_workers: 16
        sources:
        - class_name: StitchedDataset
          data_name: StitchedDolma
          class_args: {stitched_seq_path: ..., tokenized_data_root: ..., split_ratio: ...}
        - class_name: StitchedDataset
          data_name: StitchedDolma2
          class_args: {stitched_seq_path: ..., tokenized_data_root: ..., split_ratio: ...}

Note: resuming assumes an unchanged datasets config. DO NOT change the source list or the source
configs between runs, including the order of sources.
"""

from __future__ import annotations

from ...parallel import ProcessGroupManager
from ..megatron import Split
from ..stitched import OrderingStrategy, StitchedDatasetConfig, StitchedSequenceDataset
from ..stitched import build_sample_index as build_stitched_sample_index
from .dataset import MixedSequenceDataset


def _validate_sources(class_args: dict) -> list[dict]:
    """Validate the sources list in a MixedDataset config.

    Criteria:
    1. Each source must declare class_name StitchedDataset.
    2. Each source must be labeled with a unique data_name.
    3. Each source must reference a unique stitched_seq_path (no two sources may mix in the
       same underlying corpus).
    4. The global sequence_length must be set in the MixedDataset config, and individual sources
       must not set sequence_length or source_sequence_length.

    Args:
        class_args: The class_args dict from the MixedDataset config.

    Returns:
        The validated sources list unchanged; raises AssertionError if any validation fails.
    """
    sources = class_args.get("sources")
    assert isinstance(sources, list) and len(sources) > 0, "MixedDataset needs a non-empty class_args.sources list"
    assert class_args.get("sequence_length") is not None, "MixedDataset needs a global class_args.sequence_length"

    data_names = []
    stitched_seq_paths = []
    for source in sources:
        class_name = source.get("class_name")
        data_name = source.get("data_name")
        source_class_args = source.get("class_args", {})

        assert class_name == "StitchedDataset", (
            f"Unexpected source class_name ({class_name}); MixedDataset only supports "
            "StitchedDataset sources, MegatronDataset is no longer supported."
        )
        assert data_name, "Every source needs a data_name."
        assert "data_sampling_ratio" not in source, (
            f"Source {data_name}: data_sampling_ratio is not supported; "
            "mixed dataset sees one example for exactly one time in each epoch."
        )
        assert "sequence_length" not in source_class_args and "source_sequence_length" not in source_class_args, (
            f"Source {data_name}: sequence_length is global; "
            f"Do not set sequence_length or source_sequence_length in individual source configs."
        )

        data_names.append(data_name)
        stitched_seq_paths.append(source_class_args.get("stitched_seq_path"))

    assert len(set(data_names)) == len(data_names), f"Source data_names must be unique, got {data_names}"

    assert len(set(stitched_seq_paths)) == len(stitched_seq_paths), (
        f"Source stitched_seq_path values must be unique (no two sources may mix in the same "
        f"underlying corpus), got {stitched_seq_paths}"
    )

    return sources


def _build_stitched_source(class_args: dict, val_samples: int, test_samples: int, default_seed: int) -> tuple[
    StitchedSequenceDataset | None,
    StitchedSequenceDataset | None,
    StitchedSequenceDataset | None,
]:
    config = StitchedDatasetConfig(
        stitched_seq_path=class_args["stitched_seq_path"],
        tokenized_data_root=class_args["tokenized_data_root"],
        sequence_length=class_args["sequence_length"],
        ordering_strategy=OrderingStrategy(class_args.get("ordering_strategy", "as_stored")),
        seed=class_args.get("seed", default_seed),
        split_ratio=tuple(class_args.get("split_ratio", [1.0, 0.0, 0.0])),
    )

    caching_allowed = class_args.get("caching_allowed", True)

    def _build_split(split: Split, num_samples: int | None) -> StitchedSequenceDataset | None:
        if num_samples == 0 or config.split_ratio[split.value] == 0.0:
            return None
        stitched_seq, sample_index = build_stitched_sample_index(config, split, None, caching_allowed)
        return StitchedSequenceDataset(config, split, stitched_seq, sample_index, num_samples=num_samples)

    # num_samples=None serves exactly one epoch, as the mixture layer requires;
    # val/test keep the full eval budget via the dataset's internal replay.
    train_ds = _build_split(Split.train, None)
    val_ds = _build_split(Split.valid, val_samples)
    test_ds = _build_split(Split.test, test_samples)

    return train_ds, val_ds, test_ds


def build_mixed_datasets(
    class_args: dict,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    default_seed: int,
) -> tuple[
    MixedSequenceDataset | None,
    MixedSequenceDataset | None,
    MixedSequenceDataset | None,
]:
    """Build the train mixture and the mixed val/test datasets.

    Args:
        class_args: The class_args dict from the MixedDataset config.
        train_samples: The number of training samples to build for the mixture.
        val_samples: The number of validation samples to build for the mixture.
        test_samples: The number of test samples to build for the mixture.
        default_seed: The default random seed to use for the datasets.

    Returns:
        train: The training dataset (MixedSequenceDataset).
        val: The validation dataset (MixedSequenceDataset or None if no sources have validation).
        test: The test dataset (MixedSequenceDataset or None if no sources have test).
    """
    sources = _validate_sources(class_args)

    train_children = []
    val_children = []
    test_children = []
    val_data_names = []
    test_data_names = []
    data_names = []

    # Build every source on every rank, in config order: stitched builds are cheap cache
    # reads/writes, but all ranks must still walk the same build sequence deterministically.
    for source in sources:
        source_class_args = {
            **source["class_args"],
            "sequence_length": class_args["sequence_length"],
        }

        train_ds, val_ds, test_ds = _build_stitched_source(source_class_args, val_samples, test_samples, default_seed)

        train_children.append(train_ds)
        data_names.append(source["data_name"])

        # Only add to val/test lists if that split exists (non-None)
        if val_ds is not None:
            val_children.append(val_ds)
            val_data_names.append(source["data_name"])

        if test_ds is not None:
            test_children.append(test_ds)
            test_data_names.append(source["data_name"])

    # The trainer's convention is that dataloaders exist only on tensor-parallel-first
    # ranks; other ranks get None loaders and receive batches via the tensor-parallel
    # input broadcast.
    if ProcessGroupManager.is_initialized() and not ProcessGroupManager.is_tensor_parallel_first_rank():
        return None, None, None

    # Validate all sources have training data
    for data_name, train_ds in zip(data_names, train_children):
        assert train_ds is not None, f"source {data_name} has an empty train split"

    # Build training mixture (with per-epoch reshuffle)
    train_ds = MixedSequenceDataset(
        datasets=train_children,
        data_names=data_names,
        num_samples=train_samples,
        seed=class_args.get("seed", default_seed),
        reshuffle_per_epoch=True,
    )

    # Build validation mixture if any source has validation (no per-epoch reshuffle)
    val_ds = None
    if val_children:
        val_ds = MixedSequenceDataset(
            datasets=val_children,
            data_names=val_data_names,
            num_samples=val_samples,
            seed=class_args.get("seed", default_seed),
            reshuffle_per_epoch=False,  # Deterministic replay for validation
        )

    # Build test mixture if any source has test (no per-epoch reshuffle)
    test_ds = None
    if test_children:
        test_ds = MixedSequenceDataset(
            datasets=test_children,
            data_names=test_data_names,
            num_samples=test_samples,
            seed=class_args.get("seed", default_seed),
            reshuffle_per_epoch=False,  # Deterministic replay for test
        )

    return train_ds, val_ds, test_ds
