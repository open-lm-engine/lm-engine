# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
import subprocess
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from transformers import AutoTokenizer

from lm_engine.data.megatron.preprocess_data import convert_file


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
    group.add_argument("--max-processes", type=int, default=16, help="Number of processes to launch")
    args = parser.parse_args()

    return args


def process_file(args: Namespace, input_file: str, output_prefix: str):
    """Process a single file with convert_file()."""
    convert_file(
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
        input_file=input_file,
        output_prefix=output_prefix,
        subset=args.subset,
        json_keys=args.json_keys,
        append_eos_token=args.append_eod,
    )


def main() -> None:
    args = get_args()

    if os.path.isfile(args.input):
        process_file(args, args.input, args.output_prefix)
    elif os.path.isdir(args.input):
        files = []
        processes = []

        for root, _, _files in os.walk(args.input):
            for file in _files:
                output_prefix = os.path.join(args.output_prefix, root.removeprefix(args.input).lstrip(os.path.sep))
                os.makedirs(output_prefix, exist_ok=True)

                files.append(
                    (
                        os.path.join(root, file),
                        os.path.join(output_prefix, os.path.splitext(file)[0]),
                    )
                )

        for input_file, output_prefix in tqdm(files, desc="Tokenizing"):
            assert args.json_keys == ["text"]

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
            ]

            if args.subset:
                cmd += ["--subset", args.subset]
            if args.append_eod:
                cmd += ["--append-eod"]

            while len(processes) >= args.max_processes:
                # Remove finished processes
                processes = [p for p in processes if p.poll() is None]

            p = subprocess.Popen(cmd)
            processes.append(p)

        # Wait for all processes to complete
        for p in tqdm(processes, desc="Waiting for jobs"):
            p.wait()

        print("✅ All files processed successfully.")


if __name__ == "__main__":
    main()
