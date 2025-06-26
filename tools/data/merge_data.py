# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from argparse import ArgumentParser, Namespace

from lm_engine.data.megatron.merge_data import merge_files


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input-prefixes",
        type=str,
        nargs="+",
        required=True,
        help="Path to directory containing all document files to merge",
    )
    parser.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    merge_files(input_prefixes=args.input_prefixes, output_prefix=args.output_prefix)
