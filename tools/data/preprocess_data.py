# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import subprocess
import tempfile
import traceback
from argparse import ArgumentParser, Namespace
from typing import Iterator

import multistorageclient as msc
import pyarrow as pa
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_engine.data.megatron.bin import get_bin_path
from lm_engine.data.megatron.dtype import DType
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDatasetBuilder, get_bin_path, get_idx_path
from lm_engine.defaults import MSC_PREFIX
from lm_engine.logging_utils import log_rank_0, set_logger
from lm_engine.tokenizers import TOKENIZER_TYPE, get_tokenizer
from lm_engine.utils import is_ray_available, is_zstandard_available


if is_ray_available():
    import ray

    @ray.remote
    def process_file_ray(args: Namespace, input_file: str, output_prefix: str) -> None:
        """Ray remote function to process a single file."""

        try:
            if args.download_locally:
                with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
                    input_file, local_input_file = _convert_path_to_msc_path_and_tmp_path(
                        input_file, args.msc_base_path, tmpdir
                    )

                    output_prefix, local_output_prefix = _convert_path_to_msc_path_and_tmp_path(
                        output_prefix, args.msc_base_path, tmpdir
                    )

                    msc.download_file(input_file, local_input_file)

                    os.makedirs(os.path.dirname(local_output_prefix), exist_ok=True)

                    skipped_line_count = convert_file(
                        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                        input_file=local_input_file,
                        output_prefix=local_output_prefix,
                        subset=args.subset,
                        json_keys=args.json_keys,
                        append_eos_token=args.append_eod,
                    )

                    for key in args.json_keys:
                        for path_function in [get_bin_path, get_idx_path]:
                            msc.upload_file(
                                path_function(f"{output_prefix}_{key}"), path_function(f"{local_output_prefix}_{key}")
                            )
            else:
                skipped_line_count = convert_file(
                    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                    input_file=input_file,
                    output_prefix=output_prefix,
                    subset=args.subset,
                    json_keys=args.json_keys,
                    append_eos_token=args.append_eod,
                )

            return input_file, skipped_line_count
        except Exception as e:
            return "!!!!!!!!!!!!!!! Error Look Here !!!!!!!!!!!!!!!" + str(e) + "\n\n" + traceback.format_exc(), 0


if is_zstandard_available():
    from zstandard import ZstdDecompressor


set_logger()


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


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON/Arrow")
    group.add_argument(
        "--subset", type=str, default=None, help="Subset argument when loading input data from a HuggingFace dataset"
    )
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--max-local-processes", type=int, default=16, help="Number of processes to launch (used when --use-ray)"
    )
    group.add_argument("--use-ray", action="store_true", help="whether to use Ray")
    group.add_argument("--download-locally", action="store_true", help="download file locally")
    group.add_argument("--msc-base-path", type=str, help="base path for MSC")
    group.add_argument("--tmpdir", type=str, help="temporary local directory")

    args = parser.parse_args()

    if args.download_locally:
        assert args.msc_base_path
        assert args.tmpdir

    return args


def _convert_path_to_msc_path_and_tmp_path(path: str, base_msc_path: str, tmpdir: str) -> tuple[str, str]:
    path = path.lstrip(os.sep)
    _, base_path = path.split(os.sep, 1)
    path = os.path.join(base_msc_path, base_path)
    path = f"{MSC_PREFIX}{path}"
    local_path = os.path.join(tmpdir, base_path)
    return path, local_path


def collect_files(args: Namespace, makedirs: bool) -> list[tuple[str, str]]:
    """Collect all files to process from input directory or single file."""
    if os.path.isfile(args.input):
        return [(args.input, args.output_prefix)]

    files = []
    for root, _, _files in os.walk(args.input):
        for file in _files:
            if file.startswith("."):
                continue

            output_prefix = os.path.join(args.output_prefix, root.removeprefix(args.input).lstrip(os.path.sep))

            if makedirs:
                os.makedirs(output_prefix, exist_ok=True)

            output_prefix = os.path.join(output_prefix, os.path.splitext(file)[0])
            # check for .jsonl.zst
            if output_prefix.endswith(".jsonl"):
                output_prefix = os.path.splitext(output_prefix)[0]

            files.append((os.path.join(root, file), output_prefix))

    return sorted(files, key=lambda x: x[0])


def process_with_ray(args: Namespace, files: list) -> None:
    """Process files using Ray for distributed execution."""

    # Initialize Ray
    ray.init(
        address="auto",
        runtime_env={"env_vars": {"MSC_CONFIG": os.environ.get("MSC_CONFIG", "")}},
        log_to_driver=True,
    )
    log_rank_0(logging.INFO, "Ray initialized for processing.")

    # Wait for completion with progress bar
    futures = []
    for input_file, output_prefix in files:
        futures.append(
            process_file_ray.options(num_cpus=1).remote(args=args, input_file=input_file, output_prefix=output_prefix)
        )

    with tqdm(total=len(files), desc="Tokenizing") as pbar:
        # Loop until no remaining files OR futures
        while futures:
            # Wait for one task to complete
            done, futures = ray.wait(futures, num_returns=1)
            future = done[0]

            try:
                result, skipped_line_count = ray.get(future)
                if "Error" in result:
                    log_rank_0(logging.ERROR, f"❌ Error processing file: {result}")
                else:
                    log_rank_0(logging.INFO, f"✅ Processed file: {result}")
                if skipped_line_count > 0:
                    log_rank_0(logging.WARNING, f"❌ Skipped {skipped_line_count} lines for file: {result}")
            except Exception as e:
                log_rank_0(logging.ERROR, f"❌ Error processing file: {e}. {traceback.format_exc()}")

            pbar.update(1)

    ray.shutdown()


def process_with_subprocess(args: Namespace, files: list):
    """Process files using subprocess for local parallel execution."""
    log_rank_0(
        logging.INFO,
        f"🔧 Processing {len(files)} files with subprocesses (max {args.max_local_processes} parallel)",
    )

    processes = []
    for input_file, output_prefix in tqdm(files, desc="Tokenizing"):
        # Launch subprocess in background
        cmd = [
            "python",
            __file__,
            "--input",
            input_file,
            "--output-prefix",
            output_prefix,
            "--tokenizer",
            args.tokenizer,
            "--json-keys",
            *args.json_keys,
        ]

        if args.subset:
            cmd += ["--subset", args.subset]
        if args.append_eod:
            cmd += ["--append-eod"]

        while len(processes) >= args.max_local_processes:
            # Remove finished processes
            processes = [p for p in processes if p.poll() is None]

        p = subprocess.Popen(cmd)
        processes.append(p)

    # Wait for all processes to complete
    for p in tqdm(processes, desc="Waiting for jobs"):
        p.wait()


def main() -> None:
    args = get_args()

    msc_config_path = os.environ.get("MSC_CONFIG", "")
    if msc_config_path:
        assert os.path.isabs(msc_config_path)

    # Single file processing (direct call, no parallelization)
    if os.path.isfile(args.input) and not args.use_ray:
        convert_file(
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
            input_file=args.input,
            output_prefix=args.output_prefix,
            subset=args.subset,
            json_keys=args.json_keys,
            append_eos_token=args.append_eod,
        )

        log_rank_0(logging.INFO, "✅ File processed successfully.")
        return

    # Collect all files
    files = collect_files(args, makedirs=not args.download_locally)

    if not files:
        log_rank_0(logging.INFO, "❌ No files found to process")
        return

    if args.use_ray:
        assert is_ray_available()
        process_with_ray(args, files)
    else:
        process_with_subprocess(args, files)

    log_rank_0(logging.INFO, "✅ All files processed successfully.")


if __name__ == "__main__":
    main()
