# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from argparse import ArgumentParser, Namespace

from lm_engine.data.megatron.merge_data import merge_files


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input-prefixes",
        type=str,
        nargs="+",
        required=False,
        help="Path to directory containing all document files to merge",
    )

    parser.add_argument(
        "--input-directory",
        type=str,
        required=False,
        help="Path to directory containing all document files to merge",
    )

    parser.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    args = parser.parse_args()
    assert len(args.input_prefixes) > 0 or args.input_directory is not None

    return args


if __name__ == "__main__":
    args = get_args()

    if len(args.input_prefixes):
        merge_files(input_prefixes=args.input_prefixes, output_prefix=args.output_prefix)
    elif args.input_directory is not None:
        files = os.listdir(args.input_directory)
        files = list(filter(lambda x: x.endswith(".bin"), files))
        files = [i.split(".bin")[0] for i in files]

        merge_files(input_prefixes=files, output_prefix=args.output_prefix)
