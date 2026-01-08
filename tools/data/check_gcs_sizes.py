# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""
Check sizes of .bin files in GCS bucket folders.

Iterates over specified GCS paths, lists all .bin files, and reports their sizes
(in bytes and GB) grouped by folder structure. Also calculates total folder sizes.

Usage:
    python check_gcs_sizes.py gs://mayank-data/dolma1/ gs://mayank-data/dolma3-pool/ gs://mayank-data/arxiv-redpajama/ gs://mayank-data/Nemotron-CC-v2/ gs://mayank-data/finemath/ gs://mayank-data/the-stack-v2-dedup/ gs://mayank-data/stack-edu/
"""

import argparse
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


def format_size(size_bytes: int) -> str:
    """Format size in bytes.

    - If the size is over 1 TB (1024 GB), display it in TB
    - Otherwise display it in GB
    """
    GB = 1024**3
    TB = 1024**4

    if size_bytes >= TB:
        tb_size = size_bytes / TB
        return f"{size_bytes:,} bytes ({tb_size:.2f} TB)"
    else:
        gb_size = size_bytes / GB
        return f"{size_bytes:,} bytes ({gb_size:.2f} GB)"


def main():
    parser = argparse.ArgumentParser(description="Check .bin file sizes in GCS bucket folders.")
    parser.add_argument(
        "gcs_paths",
        type=str,
        nargs="+",
        help="List of GCS paths to check (e.g., gs://bucket-name/path/to/folder)",
    )
    args = parser.parse_args()

    all_paths = args.gcs_paths

    for gcs_path in all_paths:
        try:
            bucket_name, prefix = parse_gcs_path(gcs_path)
            print(f"Scanning: gs://{bucket_name}/{prefix} recursively...")
            print("=" * 80)

            grouped_files = list_files_grouped_by_dir(bucket_name, prefix)
            sorted_dirs = sorted(grouped_files.keys())

            path_total_size = 0
            path_file_count = 0

            for directory in sorted_dirs:
                files = grouped_files[directory]
                full_dir_path = f"gs://{bucket_name}/{directory}" if directory else f"gs://{bucket_name}/"

                # Filter for .bin files
                bin_files = {f: s for f, s in files.items() if f.endswith(".bin")}

                if not bin_files:
                    continue

                print(f"\n📁 {full_dir_path}")

                folder_size = 0

                # Sort numerically by the number before the extension (e.g., 2.bin, 10.bin)
                def _numeric_key(item: tuple[str, int]):
                    filename, _ = item
                    num_part = filename.split(".", 1)[0]
                    return int(num_part) if num_part.isdigit() else filename

                sorted_files = sorted(bin_files.items(), key=_numeric_key)

                for filename, size in sorted_files:
                    print(f"  - {filename:<30} : {format_size(size)}")
                    folder_size += size

                print(f"  {'-'*60}")
                print(f"  Total Folder Size              : {format_size(folder_size)}")

                path_total_size += folder_size
                path_file_count += len(bin_files)

            print("=" * 80)
            print(f"Total for {gcs_path}:")
            print(f"  Files: {path_file_count}")
            print(f"  Size : {format_size(path_total_size)}")
            print("\n")

        except Exception as e:
            print(f"Error processing {gcs_path}: {e}")
            print("\n")


if __name__ == "__main__":
    main()
