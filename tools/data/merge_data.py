# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import multiprocessing
import os
import tempfile
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

import multistorageclient as msc

from lm_engine.data.megatron.merge_data import merge_files
from lm_engine.defaults import MSC_PREFIX


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
        "--workers", type=int, default=1, required=False, help="number of workers"
    )

    parser.add_argument(
        "--input-directory",
        type=str,
        required=False,
        help="Path to directory containing all document files to merge",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    parser.add_argument("--max-size", type=int, required=False, help="max file size")

    parser.add_argument(
        "--download-locally", action="store_true", help="download file locally"
    )
    parser.add_argument("--msc-base-path", type=str, help="base path for MSC")
    parser.add_argument("--tmpdir", type=str, help="temporary local directory")

    args = parser.parse_args()
    assert args.input_prefixes is not None or args.input_directory is not None

    if args.download_locally:
        assert args.msc_base_path
        assert args.tmpdir

    return args


def _convert_path_to_msc_path_and_tmp_path(
    path: str, base_msc_path: str, tmpdir: str
) -> tuple[str, str]:
    path = path.lstrip(os.sep)
    try:
        _, base_path = path.split(os.sep, 1)
    except ValueError:
        base_path = path

    path = os.path.join(base_msc_path, base_path)
    path = f"{MSC_PREFIX}{path}"
    local_path = os.path.join(tmpdir, base_path)
    return path, local_path


def merge_files_wrapper(
    input_prefixes: list[str], output_prefix: str, args: Namespace
) -> None:
    if args.download_locally:
        with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
            local_input_prefixes = []
            for prefix in tqdm(input_prefixes, desc="Downloading files"):
                # Calculate paths for .bin and .idx
                msc_prefix_path, local_prefix_path = _convert_path_to_msc_path_and_tmp_path(
                    prefix, args.msc_base_path, tmpdir
                )

                os.makedirs(os.path.dirname(local_prefix_path), exist_ok=True)

                for ext in [".bin", ".idx"]:
                    msc.download_file(
                        f"{msc_prefix_path}{ext}", f"{local_prefix_path}{ext}"
                    )

                local_input_prefixes.append(local_prefix_path)

            # Calculate output paths
            msc_output_path, local_output_path = _convert_path_to_msc_path_and_tmp_path(
                output_prefix, args.msc_base_path, tmpdir
            )
            os.makedirs(os.path.dirname(local_output_path), exist_ok=True)

            merge_files(
                input_prefixes=local_input_prefixes, output_prefix=local_output_path
            )

            for ext in [".bin", ".idx"]:
                msc.upload_file(f"{msc_output_path}{ext}", f"{local_output_path}{ext}")

    else:
        merge_files(input_prefixes=input_prefixes, output_prefix=output_prefix)


def get_groups_by_sizes(path: str, max_size: int | None = None) -> list[list[str]]:
    # Expand path to include sibling directories with same prefix
    path = path.rstrip(os.sep)
    base_dir = os.path.dirname(path)
    # if not base_dir:
    #     base_dir = "."
    search_prefix = os.path.basename(path)

    input_dirs = []
    if os.path.isdir(base_dir):
        # We need to list directories in base_dir
        for d in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path) and d.startswith(search_prefix):
                input_dirs.append(full_path)
    if not input_dirs:
        input_dirs = [path]

    print(f"Merging content from directories: {input_dirs}")

    fnames = []
    for d in input_dirs:
        if os.path.isdir(d):
            curr_fnames = filter(lambda x: x.endswith(".bin"), os.listdir(d))
            curr_fnames = [os.path.join(d, i) for i in curr_fnames]
            fnames.extend(curr_fnames)

    fnames = [i[:-4] for i in fnames]

    if max_size is None:
        return [fnames]

    max_size *= 1024**3
    groups = []
    current_grp = []
    current_size = 0

    for index, fname in enumerate(tqdm(fnames, desc="Grouping files")):
        current_grp.append(fname)
        current_size += os.path.getsize(f"{fname}.bin")

        if current_size > max_size or index == len(fnames) - 1:
            groups.append(current_grp)
            current_grp = []
            current_size = 0

    return groups


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_prefix, exist_ok=True)

    if args.input_prefixes is not None:
        assert args.max_size is None
        merge_files_wrapper(
            input_prefixes=args.input_prefixes, output_prefix=args.output_prefix, args=args
        )
    elif args.input_directory is not None:
        file_groups = get_groups_by_sizes(args.input_directory, args.max_size)

        if args.max_size is None:
            merge_files_wrapper(
                input_prefixes=file_groups[0], output_prefix=args.output_prefix, args=args
            )
        else:
            pool = multiprocessing.Pool(args.workers)
            pbar = tqdm(total=len(file_groups), desc="Merging files")

            def update(*a):
                pbar.update()

            for grp_id, group in enumerate(file_groups):
                pool.apply_async(
                    merge_files_wrapper,
                    kwds=dict(
                        input_prefixes=group,
                        output_prefix=os.path.join(args.output_prefix, str(grp_id)),
                        args=args,
                    ),
                    callback=update,
                )

                json.dump(
                    group,
                    open(os.path.join(args.output_prefix, f"file_map-{grp_id}.json"), "w"),
                    indent=4,
                )

            pool.close()
            pool.join()
            pbar.close()
