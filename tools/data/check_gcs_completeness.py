# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""
Check file completeness in a GCS bucket folder.

For each file_map-{id}.json, verifies that corresponding {id}.bin and {id}.idx files exist.
Checks files recursively at every directory level found under the specified path.

Usage:
    python check_gcs_completeness.py gs://mayank-data/dolma1/ gs://mayank-data/dolma3-pool/ gs://mayank-data/arxiv-redpajama/ gs://mayank-data/Nemotron-CC-v2/ gs://mayank-data/finemath/ gs://mayank-data/the-stack-v2-dedup/ gs://mayank-data/stack-edu/
    python check_gcs_completeness.py gs://bucket-name/path/to/folder --verbose
"""

import argparse
import json
import re
from collections import defaultdict

from google.cloud import storage


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a GCS path into bucket name and prefix."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with 'gs://'")

    path = gcs_path[5:]  # Remove 'gs://'
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Ensure prefix ends with '/' if not empty
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return bucket_name, prefix


def list_files_grouped_by_dir(bucket_name: str, prefix: str) -> dict[str, dict[str, int]]:
    """
    List all files recursively starting from prefix, grouped by directory.
    Returns: dict[directory_path, dict[filename, size]]
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs recursively (no delimiter)
    blobs = bucket.list_blobs(prefix=prefix)

    grouped_files = defaultdict(dict)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        # Split into directory and filename
        if "/" in blob.name:
            directory, filename = blob.name.rsplit("/", 1)
            directory += "/"
        else:
            directory = ""
            filename = blob.name

        grouped_files[directory][filename] = blob.size

    return grouped_files


def check_sizes(files: dict[str, int]) -> list[str]:
    """
    Check if .bin files are within 230-290GB, excluding the last index.
    Returns a list of size violation messages.
    """
    idx_pattern = re.compile(r"^(\d+)\.bin$")
    idx_files = {}  # id -> (filename, size)

    for filename, size in files.items():
        match = idx_pattern.match(filename)
        if match:
            idx_files[int(match.group(1))] = (filename, size)

    if not idx_files:
        return []

    max_id = max(idx_files.keys())
    size_issues = []

    # 230GB - 290GB
    GB = 1024**3
    min_size = 230 * GB
    max_size = 290 * GB

    for id_, (filename, size) in sorted(idx_files.items()):
        if id_ == max_id:
            print(f"{filename}: {size / GB:.2f} GB")
            continue

        if not (min_size <= size <= max_size):
            size_issues.append(f"{filename}: {size / GB:.2f} GB (expected 230-290 GB)")
        # else:
        #     print(f"{filename}: {size / GB:.2f} GB")

    return size_issues


def check_completeness(files: list[str], verbose: bool = False) -> tuple[list[str], list[str], dict]:
    """
    Check file completeness.

    Returns:
        - List of missing files
        - List of orphan files (files without corresponding file_map)
        - Dict of complete sets
    """
    # Extract all IDs from file_map-{id}.json files
    file_map_pattern = re.compile(r"^file_map-(\d+)\.json$")
    bin_pattern = re.compile(r"^(\d+)\.bin$")
    idx_pattern = re.compile(r"^(\d+)\.idx$")

    file_map_ids = set()
    bin_ids = set()
    idx_ids = set()

    for f in files:
        match = file_map_pattern.match(f)
        if match:
            file_map_ids.add(int(match.group(1)))
            continue

        match = bin_pattern.match(f)
        if match:
            bin_ids.add(int(match.group(1)))
            continue

        match = idx_pattern.match(f)
        if match:
            idx_ids.add(int(match.group(1)))

    # Check for missing files
    missing_files = []
    complete_sets = {}

    for id_ in sorted(file_map_ids):
        missing_for_id = []
        if id_ not in bin_ids:
            missing_files.append(f"{id_}.bin")
            missing_for_id.append("bin")
        if id_ not in idx_ids:
            missing_files.append(f"{id_}.idx")
            missing_for_id.append("idx")

        if not missing_for_id:
            complete_sets[id_] = True

    # Check for orphan .bin and .idx files (no corresponding file_map)
    orphan_files = []
    all_data_ids = bin_ids | idx_ids

    for id_ in sorted(all_data_ids - file_map_ids):
        if id_ in bin_ids:
            orphan_files.append(f"{id_}.bin")
        if id_ in idx_ids:
            orphan_files.append(f"{id_}.idx")

    # Check for .bin without .idx and vice versa
    for id_ in sorted(bin_ids - idx_ids):
        if id_ in file_map_ids:
            continue  # Already reported as missing
        if f"{id_}.idx" not in orphan_files and f"{id_}.idx" not in missing_files:
            missing_files.append(f"{id_}.idx (orphan .bin exists)")

    for id_ in sorted(idx_ids - bin_ids):
        if id_ in file_map_ids:
            continue  # Already reported as missing
        if f"{id_}.bin" not in orphan_files and f"{id_}.bin" not in missing_files:
            missing_files.append(f"{id_}.bin (orphan .idx exists)")

    return missing_files, orphan_files, complete_sets


def main():
    parser = argparse.ArgumentParser(description="Check file completeness in a GCS bucket folder (recursively).")
    parser.add_argument(
        "gcs_paths",
        type=str,
        nargs="+",
        help="List of GCS paths to check (e.g., gs://bucket-name/path/to/folder)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including complete sets",
    )
    args = parser.parse_args()

    all_paths = args.gcs_paths

    overall_status = {}
    for gcs_path in all_paths:
        bucket_name, prefix = parse_gcs_path(gcs_path)
        print(f"Scanning: gs://{bucket_name}/{prefix} recursively...")
        print("-" * 60)
        grouped_files = list_files_grouped_by_dir(bucket_name, prefix)

        # Track overall status
        any_issues = False

        sorted_dirs = sorted(grouped_files.keys())

        for directory in sorted_dirs:
            files = grouped_files[directory]
            full_dir_path = f"gs://{bucket_name}/{directory}" if directory else f"gs://{bucket_name}/"

            print(f"\n📁 Checking: {full_dir_path}")
            print(f"Found {len(files)} files")

            # Count file types
            file_maps = [f for f in files if f.startswith("file_map-") and f.endswith(".json")]
            bins = [f for f in files if f.endswith(".bin")]
            idxs = [f for f in files if f.endswith(".idx")]

            print(f"  - file_map files: {len(file_maps)}")
            print(f"  - .bin files: {len(bins)}")
            print(f"  - .idx files: {len(idxs)}")

            missing_files, orphan_files, complete_sets = check_completeness(list(files.keys()), args.verbose)

            # Check sizes (prints details to stdout)
            size_issues = check_sizes(files)

            if args.verbose:
                print(f"Complete sets: {len(complete_sets)}")
                if complete_sets:
                    ids = sorted(complete_sets.keys())
                    if len(ids) > 10:
                        print(f"  IDs: {ids[:5]} ... {ids[-5:]}")
                    else:
                        print(f"  IDs: {ids}")

            if missing_files:
                print(f"❌ Missing files ({len(missing_files)}):")
                for f in missing_files:
                    print(f"  - {f}")
                any_issues = True
            else:
                print("✅ No missing files!")

            if size_issues:
                print(f"❌ Size issues ({len(size_issues)}):")
                for f in size_issues:
                    print(f"  - {f}")
                any_issues = True
            else:
                print("✅ No size issues!")

            if orphan_files:
                print(f"⚠️  Orphan files (no corresponding file_map) ({len(orphan_files)}):")
                for f in orphan_files:
                    print(f"  - {f}")
                any_issues = True

            if any_issues:
                overall_status[gcs_path] = "❌"
            # else:
            #     overall_status[gcs_path] = "✅"
            print("-" * 60)

    print(json.dumps(overall_status, indent=4))


if __name__ == "__main__":
    main()
