# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
Generate a synthetic stitched sequence file for testing the StitchedSequenceDataset.

Each entry in the stitched sequence is (shard_id, beg, end) where:
  - shard_id: name of the shard directory under data_root
  - beg, end: contiguous slice of documents within that shard [beg, end)

Output: parquet file with columns [shard_id, beg, end]
"""

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd

from lm_engine.training.data.megatron.indexed_dataset import MMapIndexedDataset


def find_shards(data_root: str) -> list[str]:
    """Return list of shard_ids as relative paths (dir/stem) for every .bin file under data_root."""
    shards = []
    for entry in sorted(os.listdir(data_root)):
        shard_dir = os.path.join(data_root, entry)
        if not os.path.isdir(shard_dir):
            continue
        for bin_file in sorted(glob.glob(os.path.join(shard_dir, "*_content.bin"))):
            stem = os.path.basename(bin_file).replace(".bin", "")
            shards.append(os.path.join(entry, stem))
    return shards


def get_shard_path_prefix(data_root: str, shard_id: str) -> str:
    """Return the full path prefix (without .bin/.idx) for a given shard_id."""
    return os.path.join(data_root, shard_id)


def sample_collection(num_docs: int, min_size: int, max_size: int) -> tuple[int, int]:
    """Sample a random [beg, end) slice from a shard with num_docs documents."""
    size = random.randint(min_size, min(max_size, num_docs))
    beg = random.randint(0, num_docs - size)
    end = beg + size
    return beg, end


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic stitched sequence")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing shard subdirectories")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--num_collections", type=int, default=1000, help="Number of collections to sample")
    parser.add_argument("--min_collection_size", type=int, default=1, help="Minimum number of docs per collection")
    parser.add_argument("--max_collection_size", type=int, default=50, help="Maximum number of docs per collection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Scanning shards in {args.data_root}...")
    shards = find_shards(args.data_root)
    print(f"Found {len(shards)} shards")

    # Load sequence lengths from each shard (just the .idx, not the .bin)
    shard_num_docs = {}
    for shard_id in shards:
        path_prefix = get_shard_path_prefix(args.data_root, shard_id)
        ds = MMapIndexedDataset(path_prefix)
        shard_num_docs[shard_id] = len(ds)
        print(f"  {shard_id}: {len(ds)} docs")

    # Sample random collections
    records = []
    sampled_shards = random.choices(shards, k=args.num_collections)
    for shard_id in sampled_shards:
        num_docs = shard_num_docs[shard_id]
        if num_docs < args.min_collection_size:
            continue
        beg, end = sample_collection(num_docs, args.min_collection_size, args.max_collection_size)
        records.append({"shard_id": shard_id, "beg": beg, "end": end})

    df = pd.DataFrame(records, columns=["shard_id", "beg", "end"])
    df["beg"] = df["beg"].astype(np.int32)
    df["end"] = df["end"].astype(np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df)} collections to {args.output}")
    print(f"Total docs: {(df['end'] - df['beg']).sum()}")
    print(f"Unique shards used: {df['shard_id'].nunique()} / {len(shards)}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
