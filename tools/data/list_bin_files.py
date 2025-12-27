# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""
List all .bin files and their sizes in a GCS bucket folder (recursively).

Usage:
    python list_bin_files.py gs://bucket-name/path/to/folder
"""

import argparse

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


def list_bin_files(gcs_path: str):
    """List .bin files and their sizes recursively."""
    bucket_name, prefix = parse_gcs_path(gcs_path)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        return

    print(f"Scanning: gs://{bucket_name}/{prefix} recursively...")
    print("-" * 60)

    # List all blobs recursively
    blobs = bucket.list_blobs(prefix=prefix)

    count = 0
    total_size = 0
    current_dir = None

    dir_count = 0
    for blob in blobs:
        if blob.name.endswith(".bin"):
            # Get directory
            if "/" in blob.name:
                directory = blob.name.rsplit("/", 1)[0]
            else:
                directory = ""

            if current_dir is not None and directory != current_dir:
                print("-" * 40)
                dir_count = 0
            dir_count += 1

            current_dir = directory

            # blob.name includes the prefix (nested directories)
            size_gb = blob.size / (1024**3)
            print(f"{blob.name}: {size_gb:.2f} GB")
            # if size_gb > 10 and dir_count != 1:
            #     print(f"!!!!!! {blob.name} is too large: {size_gb:.2f} GB")
            count += 1
            total_size += blob.size

    print("-" * 60)
    print(f"Total .bin files: {count}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="List .bin files and their sizes in a GCS bucket folder.")
    parser.add_argument(
        "gcs_path",
        nargs="?",
        default="gs://mayank-data/tmp-tokenized/the-stack-v2-dedup/",
        help="GCS path to check (default: gs://mayank-data/tmp-tokenized/the-stack-v2-dedup/)",
    )
    args = parser.parse_args()

    list_bin_files(args.gcs_path)


if __name__ == "__main__":
    main()
