<!-- **************************************************
Copyright (c) 2026, Mayank Mishra
************************************************** -->

# StitchedSequenceDataset

Produces fixed-length token sequences for pretraining from a **stitched sequence** — a parquet file that maps named document ranges into collections. Documents within a collection are concatenated into a token stream, and samples tile that stream without gaps. Document ordering within a collection is configurable.

---

## Concepts

**Shard** — a `MMapIndexedDataset` binary (`.bin` + `.idx`). Contains many documents.

**Collection** — a named slice of a shard: `(shard_id, beg, end)`. One row in the parquet.

**sample_index** — `[N+1, 3]` int32 array built once and optionally cached. Each row is a boundary `(collection_idx, doc_offset, token_offset)`. Adjacent rows define one sample.

---

## Input: stitched_seq parquet

| shard_id | beg | end |
|---|---|---|
| `cc/shard_001` | 0 | 5000 |
| `cc/shard_001` | 5000 | 9000 |
| `code/shard_042` | 0 | 3000 |

`shard_id` is a path relative to `data_root`. `beg`/`end` are doc indices within the shard.

---

## Config

```yaml
datasets:
  - class_name: StitchedDataset
    class_args:
      stitched_seq_path: /path/to/stitched_seq.parquet
      data_root: /path/to/tokenized/data
      sequence_length: 4096
      data_cache_path: /path/to/cache        # optional, caches sample_index on disk
      ordering_strategy: sequential           # or: random
      split_ratio: [0.99, 0.01, 0.0]         # train / val / test over collections
      eval_steps: 10
      num_workers: 4
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | `StitchedDatasetConfig` dataclass + `OrderingStrategy` enum |
| `utils.py` | `Split` IntEnum (train=0, val=1, test=2) |
| `builder.py` | Builds or loads cached `sample_index` from the parquet + `.idx` files |
| `ordering.py` | `get_doc_order` — returns doc indices sequential or shuffled |
| `shard_store.py` | LRU cache of open `MMapIndexedDataset` objects, one per DataLoader worker |
| `dataset.py` | `StitchedSequenceDataset` — `__getitem__` reads tokens via `ShardStore` |
| `__init__.py` | `get_stitched_dataloaders` — entry point, mirrors `get_megatron_gpt_dataloaders` |

---

## Tests

```bash
pytest tests/training/stitched_dataset_test.py -v
```
