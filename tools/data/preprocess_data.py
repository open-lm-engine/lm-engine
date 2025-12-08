# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
import os
import subprocess
import tempfile
from argparse import ArgumentParser, Namespace
from collections import deque

import multistorageclient as msc
import ray
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_engine.data.megatron.indexed_dataset import get_bin_path, get_idx_path
from lm_engine.data.megatron.preprocess_data import convert_file
from lm_engine.defaults import MSC_PREFIX
from lm_engine.utils import log_rank_0, set_logger


set_logger()


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
        "--max-local-processes", type=int, default=16, help="Number of processes to launch (used when ray-workers=0)"
    )
    group.add_argument("--ray-workers", type=int, default=0, help="Number of ray workers (0 = use subprocess)")
    group.add_argument("--download-locally", action="store_true", help="download file locally")

    args = parser.parse_args()

    return args


def _convert_path_to_msc_path(path: str) -> str:
    path = path.lstrip(os.sep)
    _, path = path.split(os.sep, 1)
    return path


@ray.remote
def process_file_ray(args: Namespace, input_file: str, output_prefix: str) -> None:
    """Ray remote function to process a single file."""

    if args.download_locally:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_rank_0(logging.DEBUG, f"DEBUG: Using {tmpdir} as the temporary directory")

            input_file = _convert_path_to_msc_path(input_file)
            output_prefix = _convert_path_to_msc_path(output_prefix)

            local_input_file = os.path.join(tmpdir, input_file)
            local_output_prefix = os.path.join(tmpdir, output_prefix)

            log_rank_0(logging.DEBUG, f"DEBUG: input_file {input_file} corresponds to {local_input_file} locally")
            log_rank_0(
                logging.DEBUG, f"DEBUG: output_prefix {output_prefix} corresponds to {local_output_prefix} locally"
            )

            msc.download_file("msc://mayank-data/tmp/test/part_000045.parquet", local_input_file)

            convert_file(
                tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
                input_file=local_input_file,
                output_prefix=local_output_prefix,
                subset=args.subset,
                json_keys=args.json_keys,
                append_eos_token=args.append_eod,
            )

            msc.upload_file(get_bin_path(f"{MSC_PREFIX}mayank-{output_prefix[1:]}"), get_bin_path(local_output_prefix))
            msc.upload_file(get_idx_path(f"{MSC_PREFIX}mayank-{output_prefix[1:]}"), get_idx_path(local_output_prefix))
    else:
        convert_file(
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
            input_file=input_file,
            output_prefix=output_prefix,
            subset=args.subset,
            json_keys=args.json_keys,
            append_eos_token=args.append_eod,
        )

    return input_file


def collect_files(args: Namespace):
    """Collect all files to process from input directory or single file."""
    if os.path.isfile(args.input):
        return [(args.input, args.output_prefix)]

    files = []
    for root, _, _files in os.walk(args.input):
        for file in _files:
            output_prefix = os.path.join(args.output_prefix, root.removeprefix(args.input).lstrip(os.path.sep))
            os.makedirs(output_prefix, exist_ok=True)

            output_prefix = os.path.join(output_prefix, os.path.splitext(file)[0])
            # check for .jsonl.zstd
            if output_prefix.endswith(".jsonl"):
                output_prefix = os.path.splitext(output_prefix)[0]

            files.append((os.path.join(root, file), output_prefix))

    return sorted(files, key=lambda x: x[0])


def process_with_ray(args: Namespace, files: list) -> None:
    """Process files using Ray for distributed execution."""
    log_rank_0(logging.INFO, f"🚀 Processing {len(files)} files with Ray ({args.ray_workers} workers)")

    # Initialize Ray
    ray.init(runtime_env={"env_vars": {"MSC_CONFIG": os.environ.get("MSC_CONFIG", "")}})
    log_rank_0(logging.INFO, "Ray initialized for processing.")

    len(files)
    futures = []

    # Wait for completion with progress bar
    queue = deque(files)
    futures = []

    with tqdm(total=len(files), desc="Tokenizing") as pbar:
        # Loop until no remaining files OR futures
        while queue or futures:
            # Fill up the worker slots
            while queue and len(futures) < args.ray_workers:
                input_file, output_prefix = queue.popleft()
                log_rank_0(
                    logging.INFO,
                    f"DEBUG: Submitting task for {input_file}. Active futures: {len(futures)}",
                )
                futures.append(process_file_ray.remote(args=args, input_file=input_file, output_prefix=output_prefix))

            log_rank_0(logging.INFO, f"DEBUG: Waiting for tasks. active futures: {len(futures)}")
            # Wait for one task to complete
            done, futures = ray.wait(futures, num_returns=1)
            log_rank_0(
                logging.INFO,
                f"DEBUG: Task completed. done: {len(done)}, pending: {len(futures)}",
            )
            future = done[0]

            try:
                result = ray.get(future)
                log_rank_0(logging.INFO, f"✅ Processed file: {result}")
            except Exception as e:
                log_rank_0(logging.ERROR, f"❌ Error processing file: {e}")

            pbar.update(1)

    ray.shutdown()


def process_with_subprocess(args: Namespace, files: list):
    """Process files using subprocess for local parallel execution."""
    log_rank_0(
        logging.INFO, f"🔧 Processing {len(files)} files with subprocesses (max {args.max_local_processes} parallel)"
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
    if os.path.isfile(args.input) and args.ray_workers == 0:
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
    files = collect_files(args)

    if not files:
        log_rank_0(logging.INFO, "❌ No files found to process")
        return

    (process_with_ray if args.ray_workers > 0 else process_with_subprocess)(args, files)

    log_rank_0(logging.INFO, "✅ All files processed successfully.")


if __name__ == "__main__":
    main()
