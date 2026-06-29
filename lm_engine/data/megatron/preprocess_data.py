# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import gzip
import io
import json
import os
import time
from typing import Iterator

import pyarrow as pa
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from ...tokenizers import TOKENIZER_TYPE, get_tokenizer
from ...utils import is_zstandard_available
from .indexed_dataset import DType, MMapIndexedDatasetBuilder, get_bin_path, get_idx_path


if is_zstandard_available():
    from zstandard import ZstdDecompressor


class ArrowIterator:
    def __init__(self, filename: str) -> ArrowIterator:
        self.fin = pa.ipc.open_file(filename)
        self.num_records = self.fin.num_record_batches

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_records):
            doc = self.fin.get_batch(i)["tokens"].to_numpy().tolist()
            yield doc


class Encoder:
    def __init__(self, tokenizer: TOKENIZER_TYPE | str, json_keys: list[str], append_eod: bool) -> Encoder:
        self.tokenizer = get_tokenizer(AutoTokenizer.__name__, tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data) -> dict:
        ids = {}
        for key in self.json_keys:
            text = data[key]
            document_ids = self.tokenizer.encode(text)
            if len(document_ids) > 0:
                if self.append_eod:
                    document_ids.append(self.tokenizer.eos_token_id)
                ids[key] = document_ids
        return ids

    def encode(self, json_line) -> dict:
        """Safely encode a JSONL text line.

        If the line cannot be parsed as JSON or does not contain the expected
        keys, we return an **empty dictionary** so that the downstream loop
        silently skips the document instead of raising and aborting the entire
        preprocessing run. This provides resilience against occasional
        corrupted records that sometimes appear in large-scale datasets.
        """

        try:
            data = json.loads(json_line)
            return self._encode_data(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted JSON or missing fields – skip this line.
            return {}

    def encode_jsonl_zstd(self, bytes_obj) -> dict:
        try:
            json_str = bytes_obj.decode("utf-8")
        except UnicodeDecodeError:
            # Skip if the bytes cannot be decoded.
            return {}

        return self.encode(json_str)

    def encode_hf(self, sample) -> dict:
        return self._encode_data(sample)

    def convert_fms_arrow_to_megatron(self, sample) -> dict:
        if len(sample) > 0 and self.append_eod:
            sample.append(self.tokenizer.eos_token_id)

        return {"text": [sample]}


def convert_file(
    tokenizer: TOKENIZER_TYPE | str,
    input_file: str,
    output_prefix: str,
    subset: str | None = None,
    json_keys: list[str] = ["text"],
    append_eos_token: bool = True,
) -> int:
    encoder = Encoder(tokenizer, json_keys, append_eos_token)

    if input_file.endswith(".jsonl"):
        assert subset is None, f"jsonl doesn't support a subset"
        encoded_docs = map(encoder.encode, open(input_file, "r", encoding="utf-8"))
    elif input_file.endswith(".jsonl.zst"):
        assert subset is None, "zst jsonl doesn't support a subset"

        # Use a generator to stream lines and ensure the file is closed properly
        def zstd_iterator(path):
            with open(path, "rb") as compressed:
                dctx = ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    # Use a large buffer (64MB) to ensure efficient reading of very long lines
                    buffered = io.BufferedReader(reader, buffer_size=64 * 1024 * 1024)
                    for line in buffered:
                        yield line

        encoded_docs = map(encoder.encode_jsonl_zstd, zstd_iterator(input_file))
    elif input_file.endswith(".json.gz"):
        assert subset is None, "json.gz doesn't support a subset"
        encoded_docs = map(encoder.encode, gzip.open(input_file, "rt", encoding="utf-8"))
    elif input_file.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(input_file)

        def parquet_iterator():
            for batch in parquet_file.iter_batches(columns=json_keys, batch_size=10000):
                # to_pylist() is much faster than to_pandas() + row iteration
                for row in batch.to_pylist():
                    yield row

        encoded_docs = map(encoder.encode_hf, parquet_iterator())
    elif input_file.endswith(".arrow"):
        assert subset is None, f"arrow doesn't support a subset"
        encoded_docs = map(encoder.convert_fms_arrow_to_megatron, ArrowIterator(input_file))
    else:
        ds = load_dataset(input_file, use_auth_token=True, streaming=True, split="train", data_dir=subset)
        encoded_docs = map(encoder.encode_hf, ds)

    builders = {
        key: MMapIndexedDatasetBuilder(
            get_bin_path(f"{output_prefix}_{key}"), dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )
        for key in json_keys
    }

    skipped = 0

    for item in encoded_docs:
        # When the encoder fails to parse a line, it returns an empty dict. Count & skip.
        if not item:
            skipped += 1
            continue

        for key, document in item.items():
            builders[key].add_item(torch.IntTensor(document))
            builders[key].end_document()

    for key in json_keys:
        builders[key].finalize(get_idx_path(f"{output_prefix}_{key}"))

    return skipped


def convert_file_with_meta(
    tokenizer,
    input_file: str,
    output_prefix: str,
    subset: str | None = None,
    json_keys: list[str] = ["text"],
    append_eos_token: bool = True,
    meta_keys: list[str] | str | None = "all",  # SIDECAR: only addition to the signature
) -> int:
    """Tokenize a file and optionally write a per-document metadata sidecar.

    This function is identical to convert_file() when meta_keys is None.
    When meta_keys is provided, it additionally writes one .meta.jsonl file per
    json_key: document N in {output_prefix}_{key}.idx corresponds to line N in
    {output_prefix}_{key}.meta.jsonl.

    Sidecar generation is supported for .jsonl, .jsonl.zst, and .json.gz
    inputs (formats that carry structured per-document JSON).  For parquet,
    arrow, and HuggingFace dataset inputs the sidecar is silently skipped
    because the metadata fields are not available at that stage.

    Parameters
    ----------
    tokenizer, input_file, output_prefix, subset, json_keys, append_eos_token:
        Same as convert_file().
    meta_keys:
        Controls which fields are written to the sidecar:
        - "all" (default): all fields except the tokenized json_keys.
        - list[str]: specific fields to include (missing ones written as null).
        - None: no sidecar is written; falls back to plain convert_file().

    Returns
    -------
    int
        Number of skipped lines — same as convert_file().
    """

    # ── Formats that don't carry per-document JSON metadata ────────────────────
    # SIDECAR: for these formats we fall back to the original convert_file() so
    # that all non-sidecar behaviour is identical and tested.
    sidecar_supported = meta_keys is not None and (  # None → fall back to convert_file(), no sidecar
        input_file.endswith(".jsonl") or input_file.endswith(".jsonl.zst") or input_file.endswith(".json.gz")
    )
    if not sidecar_supported:
        return convert_file(
            tokenizer,
            input_file,
            output_prefix,
            subset=subset,
            json_keys=json_keys,
            append_eos_token=append_eos_token,
        )

    encoder = Encoder(tokenizer, json_keys, append_eos_token)

    # ── Raw-line iterators (mirror convert_file() exactly) ─────────────────────
    # SIDECAR: convert_file() uses map(encoder.encode, lines) which calls
    # json.loads() internally but discards the parsed dict.  We iterate raw
    # lines ourselves so we can extract metadata from `data` without parsing
    # twice.  No efficiency difference — both are lazy one-line-at-a-time.
    if input_file.endswith(".jsonl"):
        assert subset is None, "jsonl doesn't support a subset"

        def _jsonl_lines(path):
            with open(path, "r", encoding="utf-8") as f:
                yield from f

        lines = _jsonl_lines(input_file)

    elif input_file.endswith(".jsonl.zst"):
        assert subset is None, "zst jsonl doesn't support a subset"

        def _zstd_lines(path):
            with open(path, "rb") as compressed:
                dctx = ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    # 64 MB buffer — matches the value in convert_file().
                    buffered = io.BufferedReader(reader, buffer_size=64 * 1024 * 1024)
                    for line in buffered:
                        yield line.decode("utf-8", errors="replace")

        lines = _zstd_lines(input_file)

    elif input_file.endswith(".json.gz"):
        assert subset is None, "json.gz doesn't support a subset"

        def _gzip_lines(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                yield from f

        lines = _gzip_lines(input_file)

    # ── Builders — identical to convert_file() ─────────────────────────────────
    builders = {
        key: MMapIndexedDatasetBuilder(
            get_bin_path(f"{output_prefix}_{key}"),
            dtype=DType.optimal_dtype(tokenizer.vocab_size),
        )
        for key in json_keys
    }

    # SIDECAR: open one .tmp sidecar per key before the loop.
    # Writing to .tmp + atomic rename at the end ensures the final .meta.jsonl
    # only appears once finalize() has succeeded — never in a partial state.
    meta_tmp = {key: f"{output_prefix}_{key}.meta.jsonl.tmp" for key in json_keys}
    meta_final = {key: f"{output_prefix}_{key}.meta.jsonl" for key in json_keys}
    meta_files = {key: open(meta_tmp[key], "w", encoding="utf-8") for key in json_keys}

    skipped = 0
    fname = os.path.basename(input_file)
    processed = 0
    _log_interval = 100_000
    _next_log = _log_interval
    _t0 = time.time()

    try:
        for raw in lines:
            # SIDECAR: convert_file() calls encoder.encode(raw) which does
            # json.loads() + _encode_data() but returns only token ids.  We
            # split the two steps to keep `data` for metadata extraction.
            # The skip logic (empty dict → increment skipped) is identical.
            try:
                data = json.loads(raw)
                item = encoder._encode_data(data)
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped += 1
                continue

            if not item:  # identical to convert_file()
                skipped += 1
                continue

            # SIDECAR: extract metadata once per document, shared across all keys.
            # "all" → every field except the tokenized json_keys.
            # list  → only the specified fields (missing ones become null).
            if meta_keys == "all":
                meta = {k: v for k, v in data.items() if k not in json_keys}
            else:
                meta = {k: data.get(k) for k in meta_keys}

            for key, document in item.items():
                builders[key].add_item(torch.IntTensor(document))  # identical
                builders[key].end_document()  # identical
                # SIDECAR: one sidecar line per (document, key) — keeps the
                # .meta.jsonl in exact 1:1 correspondence with the .idx.
                meta_files[key].write(json.dumps(meta, ensure_ascii=False) + "\n")

            processed += 1
            if processed >= _next_log:
                elapsed = time.time() - _t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[worker] {fname}: {processed:,} lines  ({rate:,.0f} lines/s)", flush=True)
                _next_log += _log_interval

    finally:
        # SIDECAR: always close .tmp files.  If we crash mid-loop, .tmp is left
        # behind but .meta.jsonl is never written — consistent with .idx also
        # being absent (finalize() below would not have run).
        for key in json_keys:
            meta_files[key].close()

    elapsed = time.time() - _t0
    print(f"[worker] {fname}: done — {processed:,} lines, {skipped:,} skipped, {elapsed:.1f}s", flush=True)

    # ── Finalize — identical to convert_file() ─────────────────────────────────
    for key in json_keys:
        builders[key].finalize(get_idx_path(f"{output_prefix}_{key}"))
        # SIDECAR: atomic rename — sidecar appears at the same moment as .idx.
        os.replace(meta_tmp[key], meta_final[key])

    return skipped
