# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os
import tempfile
import traceback
from argparse import ArgumentParser, Namespace
from collections import deque

import multistorageclient as msc
from google.cloud import storage
from tqdm import tqdm

from lm_engine.data.megatron.merge_data import merge_files
from lm_engine.defaults import MSC_PREFIX
from lm_engine.utils import is_ray_available


if is_ray_available():
    import ray

    @ray.remote
    def merge_files_remote(subdir: str, input_prefixes: list[str], output_prefix: str, args: Namespace) -> None:
        try:
            merge_files_wrapper(subdir, input_prefixes, output_prefix, args)
        except Exception as e:
            print(f"Error processing {output_prefix}: {e}")
            traceback.print_exc()
            raise e


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input-prefixes",
        type=str,
        nargs="+",
        required=False,
        help="Path to directory containing all document files to merge",
    )

    parser.add_argument("--workers", type=int, default=1, required=False, help="number of workers")

    parser.add_argument(
        "--input-directory", type=str, required=False, help="Path to directory containing all document files to merge"
    )

    parser.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")
    parser.add_argument("--max-size", type=int, required=False, help="max file size")
    parser.add_argument("--specific-group", type=int, required=False, help="Run only specific group index")

    parser.add_argument("--download-locally", action="store_true", help="download file locally")
    parser.add_argument("--msc-base-path", type=str, help="base path for MSC")
    parser.add_argument("--tmpdir", type=str, help="temporary local directory")

    parser.add_argument("--use-ray", action="store_true", help="whether to use ray")

    args = parser.parse_args()
    assert args.input_prefixes is not None or args.input_directory is not None

    if args.download_locally:
        assert args.use_ray
        assert args.msc_base_path
        assert args.tmpdir

    if args.msc_base_path:
        assert args.use_ray

    return args


def _convert_path_to_msc_path_and_tmp_path(path: str, base_msc_path: str, tmpdir: str) -> tuple[str, str]:
    path = path.lstrip(os.sep)

    try:
        _, base_path = path.split(os.sep, 1)
    except ValueError:
        base_path = path

    path = os.path.join(base_msc_path, base_path)
    path = f"{MSC_PREFIX}{path}"
    local_path = os.path.join(tmpdir, base_path)

    return path, local_path


def _gcs_file_exists(msc_path: str) -> bool:
    """Check if a file exists in GCS given an MSC path."""
    if not msc_path.startswith(MSC_PREFIX):
        return False

    path_without_prefix = msc_path[len(MSC_PREFIX) :]
    try:
        bucket_name, blob_name = path_without_prefix.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception:
        return False


def merge_files_wrapper(subdir: str, input_prefixes: list[str], output_prefix: str, args: Namespace) -> None:
    if args.download_locally:
        with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
            local_input_prefixes = []
            # Calculate output paths
            msc_output_path, local_output_path = _convert_path_to_msc_path_and_tmp_path(
                output_prefix, args.msc_base_path, tmpdir
            )
            if _gcs_file_exists(f"{msc_output_path}.idx") and _gcs_file_exists(f"{msc_output_path}.bin"):
                print(f"File {msc_output_path} already exists, skipping download.")
                return
            for prefix in tqdm(input_prefixes, desc="Downloading files"):
                prefix = os.path.join(subdir, prefix)
                # Calculate paths for .bin and .idx
                msc_prefix_path, local_prefix_path = _convert_path_to_msc_path_and_tmp_path(
                    prefix, args.msc_base_path, tmpdir
                )

                os.makedirs(os.path.dirname(local_prefix_path), exist_ok=True)

                for ext in [".bin", ".idx"]:
                    msc.download_file(f"{msc_prefix_path}{ext}", f"{local_prefix_path}{ext}")

                local_input_prefixes.append(local_prefix_path)

            os.makedirs(os.path.dirname(local_output_path), exist_ok=True)

            merge_files(input_prefixes=local_input_prefixes, output_prefix=local_output_path)

            for ext in [".bin", ".idx"]:
                remote_path = f"{msc_output_path}{ext}"
                msc.upload_file(remote_path, f"{local_output_path}{ext}")
    else:
        merge_files(input_prefixes=input_prefixes, output_prefix=output_prefix)


def get_groups_by_sizes(path: str, max_size: int | None = None) -> list[list[str]]:
    """
    Get groups of files to merge. Each subdirectory in path is treated as its own group.

    Args:
        path: Path to directory containing subdirectories with .bin/.idx files
        max_size: Maximum size per group in GB (optional)

    Returns:
        List of groups, where each group is a list of file prefixes to merge
    """

    groups = []

    for root, _, fnames in os.walk(path):
        # Get all .bin files in this subdirectory
        fnames = filter(lambda x: x.endswith(".bin"), fnames)
        fnames = [os.path.join(root, i) for i in fnames]

        if not fnames:
            continue

        # Remove .bin extension to get prefixes
        fnames = sorted([i[:-4] for i in fnames])
        curr_subdir_groups = []

        if max_size is None:
            # All files in this subdir form one group
            curr_subdir_groups.append(fnames)
        else:
            # Split files by size
            max_size_bytes = max_size * 1024**3
            current_grp = []
            current_size = 0

            for index, fname in enumerate(fnames):
                current_grp.append(fname)
                current_size += os.path.getsize(f"{fname}.bin")

                if current_size > max_size_bytes or index == len(fnames) - 1:
                    curr_subdir_groups.append(current_grp)
                    current_grp = []
                    current_size = 0

        groups.append((root, curr_subdir_groups))

    return groups


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_prefix, exist_ok=True)

    if args.input_prefixes is not None:
        assert args.max_size is None
        assert not args.use_ray

        merge_files_wrapper(input_prefixes=args.input_prefixes, output_prefix=args.output_prefix, args=args)
    elif args.input_directory is not None:
        file_groups = get_groups_by_sizes(args.input_directory, args.max_size)

        queue = deque()
        for subdir, subdir_groups in file_groups:
            for grp_id, group in enumerate(subdir_groups):
                if args.specific_group is not None and grp_id != args.specific_group:
                    continue

                os.makedirs(os.path.join(args.output_prefix, subdir), exist_ok=True)
                json.dump(
                    {"subdir": subdir, "grp_id": grp_id, "group": group},
                    open(
                        os.path.join(args.output_prefix, subdir, f"file_map-{grp_id}.json"),
                        "w",
                    ),
                    indent=4,
                )
                queue.append((subdir, grp_id, group))

        pbar = tqdm(total=len(queue), desc="Merging files")

        if args.use_ray:
            futures = []
            ray.init(address="auto", runtime_env={"env_vars": {"MSC_CONFIG": os.environ.get("MSC_CONFIG", "")}})

            while queue or futures:
                # Submit tasks up to args.workers
                while queue and len(futures) < args.workers:
                    subdir, grp_id, group = queue.popleft()
                    output_prefix = os.path.join(args.output_prefix, subdir, str(grp_id))

                    futures.append(
                        merge_files_remote.remote(
                            subdir=subdir, input_prefixes=group, output_prefix=output_prefix, args=args
                        )
                    )

                if futures:
                    done, futures = ray.wait(futures, num_returns=1)
                    try:
                        ray.get(done[0])
                    except Exception as e:
                        print(f"Task failed: {e}")
                    pbar.update(1)

            ray.shutdown()
        else:
            while queue:
                subdir, grp_id, group = queue.popleft()
                output_prefix = os.path.join(args.output_prefix, subdir, str(grp_id))
                merge_files_wrapper(subdir=subdir, input_prefixes=group, output_prefix=output_prefix, args=args)
                pbar.update(1)

        pbar.close()
