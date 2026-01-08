# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
import os
import subprocess
import tempfile
import traceback
from argparse import ArgumentParser, Namespace

import multistorageclient as msc
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_engine.data.megatron.indexed_dataset import get_bin_path, get_idx_path
from lm_engine.data.megatron.preprocess_data import convert_file
from lm_engine.defaults import MSC_PREFIX
from lm_engine.utils import is_ray_available, log_rank_0, set_logger


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
