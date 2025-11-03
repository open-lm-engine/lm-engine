# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
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
    parser.add_argument("--max-size", type=int, required=False, help="max file size")

    args = parser.parse_args()
    assert args.input_prefixes is not None or args.input_directory is not None

    return args


def get_groups_by_sizes(path: str, max_size: int | None = None) -> list[list[str]]:
    fnames = filter(lambda x: x.endswith(".bin"), os.listdir(path))
    fnames = [os.path.join(path, i) for i in fnames]
    fnames = [i.split(".bin")[0] for i in fnames]

    if max_size is None:
        return [fnames]

    max_size *= 1024**3
    groups = []
    current_grp = []
    current_size = 0

    for index, fname in enumerate(fnames):
        current_grp.append(fname)
        current_size += os.path.getsize(f"{fname}.bin")

        if current_size > max_size or index == len(fnames) - 1:
            groups.append(current_grp)
            current_grp = []
            current_size = 0

    return groups


if __name__ == "__main__":
    args = get_args()

    if args.input_prefixes is not None:
        assert args.max_size is None
        merge_files(input_prefixes=args.input_prefixes, output_prefix=args.output_prefix)
    elif args.input_directory is not None:
        file_groups = get_groups_by_sizes(args.input_directory, args.max_size)

        if args.max_size is None:
            merge_files(input_prefixes=file_groups[0], output_prefix=args.output_prefix)
        else:
            file_map = {}

            for grp_id, group in enumerate(file_groups):
                file_map[grp_id] = group
                merge_files(input_prefixes=group, output_prefix=os.path.join(args.output_prefix, str(grp_id)))

            json.dump(file_map, open(os.path.join(args.output_prefix, "file_map.json"), "w"), indent=4)
