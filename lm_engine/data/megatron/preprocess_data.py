# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import gzip
import io
import json
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
