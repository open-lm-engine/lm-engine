# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import multiprocessing
import tempfile

import pyarrow as pa
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils import is_zstandard_available
from .indexed_dataset import DType, MMapIndexedDatasetBuilder


if is_zstandard_available():
    from zstandard import ZstdDecompressor


class ArrowIterator:
    def __init__(self, filename: str) -> None:
        self.fin = pa.ipc.open_file(filename)
        self.num_records = self.fin.num_record_batches

    def __iter__(self) -> list[int]:
        for i in range(self.num_records):
            doc = self.fin.get_batch(i)["tokens"].to_numpy().tolist()
            yield doc


class Encoder:
    def __init__(self, tokenizer: AutoTokenizer | str, json_keys: list[str], append_eod: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data):
        ids = {}
        for key in self.json_keys:
            text = data[key]
            document_ids = self.tokenizer.encode(text)
            if len(document_ids) > 0:
                if self.append_eod:
                    document_ids.append(self.tokenizer.eos_token_id)
                ids[key] = document_ids
        return ids

    def encode(self, json_line):
        data = json.loads(json_line)
        return self._encode_data(data)

    def encode_jsonl_zstd(self, bytes_obj):
        json_str = bytes_obj.decode("utf-8")
        return self.encode(json_str)

    def encode_hf(self, sample):
        return self._encode_data(sample)

    def convert_fms_arrow_to_megatron(self, sample):
        if len(sample) > 0 and self.append_eod:
            sample.append(self.tokenizer.eos_token_id)

        return {"text": [sample]}


def convert_file(
    tokenizer: AutoTokenizer | str,
    input_file: str,
    output_prefix: str,
    workers: int,
    chunk_size: int,
    subset: str | None = None,
    json_keys: list[str] = ["text"],
    append_eos_token: bool = True,
) -> None:
    encoder = Encoder(tokenizer, json_keys, append_eos_token)
    pool = multiprocessing.Pool(workers)

    if input_file.endswith(".jsonl"):
        assert subset is None, f"jsonl doesn't support a subset"
        encoded_docs = pool.imap(encoder.encode, open(input, "r", encoding="utf-8"), chunk_size)
    elif input_file.endswith(".jsonl.zst"):
        assert subset is None, f"zst jsonl doesn't support a subset"

        dctx = ZstdDecompressor()
        outfile = tempfile.TemporaryFile(suffix=input_file.rstrip(".zstd"))
        with open(input_file, "rb") as infile:
            dctx.copy_stream(infile, outfile)
        outfile.seek(0)

        encoded_docs = pool.imap(encoder.encode_jsonl_zstd, outfile, chunk_size)
    elif input_file.endswith(".arrow"):
        assert subset is None, f"arrow doesn't support a subset"
        encoded_docs = pool.imap(encoder.convert_fms_arrow_to_megatron, ArrowIterator(input_file), chunk_size)
    else:
        ds = load_dataset(input_file, use_auth_token=True, streaming=True, split="train", data_dir=subset)
        encoded_docs = pool.imap(encoder.encode_hf, ds, chunk_size)

    builders = {
        key: MMapIndexedDatasetBuilder(f"{output_prefix}_{key}.bin", dtype=DType.optimal_dtype(tokenizer.vocab_size))
        for key in json_keys
    }

    for item in encoded_docs:
        for key, document in item.items():
            builders[key].add_item(torch.IntTensor(document))
            builders[key].end_document()

    for key in json_keys:
        builders[key].finalize(f"{output_prefix}_{key}.idx")
