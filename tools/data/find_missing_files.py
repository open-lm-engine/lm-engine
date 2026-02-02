# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""
Script to find which input files are missing from the output directory.
Useful for finding failed tokenization jobs.

Supports two modes:
1. Local/mounted filesystem (default) - uses os.walk
2. GCS via google-cloud-storage - use --gcs flag

Usage (local/mounted):
    python tools/data/find_missing_files.py \
        --input /data/tmp-hans/the-stack-v2-dedup-expanded/data \
        --output /data/tmp-tokenized/the-stack-v2-dedup

Usage (GCS):
    python tools/data/find_missing_files.py \
        --input gs://mayank-data/tmp-hans/the-stack-v2-dedup-expanded/data \
        --output gs://mayank-data/tmp-tokenized/the-stack-v2-dedup \
        --gcs
"""

import os
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser(description="Find missing output files from tokenization jobs")
    parser.add_argument("--input", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--json-key", type=str, default="text", help="JSON key used for output files")
    parser.add_argument("--gcs", action="store_true", help="Use GCS directly instead of local filesystem")
    return parser.parse_args()


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a GCS path into bucket name and prefix."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with 'gs://'")

    path = gcs_path[5:]  # Remove 'gs://'
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def list_files_gcs(gcs_path: str) -> set[str]:
    """List all files recursively from a GCS path."""
    from google.cloud import storage

    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = set()
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith("/"):
            files.add(f"gs://{bucket_name}/{blob.name}")

    return files


def list_files_local(path: str) -> set[str]:
    """List all files recursively from a local path."""
    files = set()
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.startswith("."):
                files.add(os.path.join(root, filename))
    return files


def get_expected_output_prefix(input_file: str, input_base: str, output_base: str) -> str:
    """
    Given an input file path, compute the expected output prefix.
    Mirrors the logic from preprocess_data.py
    """
    # Get the relative path from input base
    rel_path = input_file.removeprefix(input_base).lstrip("/")

    # Get directory and filename
    dirname = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)

    # Remove extension(s)
    output_name = os.path.splitext(filename)[0]
    # Handle .jsonl.zst case
    if output_name.endswith(".jsonl"):
        output_name = os.path.splitext(output_name)[0]

    # Construct output prefix
    if dirname:
        return f"{output_base}/{dirname}/{output_name}"
    return f"{output_base}/{output_name}"


def main() -> None:
    args = get_args()

    input_base = args.input.rstrip("/")
    output_base = args.output.rstrip("/")

    print(f"Input path: {input_base}")
    print(f"Output path: {output_base}")
    print(f"Mode: {'GCS' if args.gcs else 'Local/Mounted filesystem'}")

    # List all input files
    print("\nListing input files...")
    if args.gcs:
        input_files = list_files_gcs(input_base)
    else:
        input_files = list_files_local(input_base)
    print(f"Found {len(input_files)} input files")

    # List all output files
    print("\nListing output files...")
    if args.gcs:
        output_files = list_files_gcs(output_base)
    else:
        output_files = list_files_local(output_base)
    print(f"Found {len(output_files)} output files")

    # Check which input files don't have corresponding output
    print("\nChecking for missing outputs...")
    missing_files = []

    for input_file in sorted(input_files):
        # Skip hidden files and Jupyter Notebooks (as per preprocess_data.py)
        basename = os.path.basename(input_file)
        if basename.startswith("."):
            continue
        if "Jupyter_Notebook" in input_file:
            continue

        # Compute expected output files
        expected_output_prefix = get_expected_output_prefix(input_file, input_base, output_base)
        expected_bin = f"{expected_output_prefix}_{args.json_key}.bin"
        expected_idx = f"{expected_output_prefix}_{args.json_key}.idx"

        # Check if both output files exist
        if expected_bin not in output_files or expected_idx not in output_files:
            missing_files.append(
                {
                    "input": input_file,
                    "expected_bin": expected_bin,
                    "expected_idx": expected_idx,
                    "bin_exists": expected_bin in output_files,
                    "idx_exists": expected_idx in output_files,
                }
            )

    if missing_files:
        print(f"\n❌ Found {len(missing_files)} missing output file(s):")
        for item in missing_files:
            print(f"\n  Input: {item['input']}")
            print(f"  Expected bin: {item['expected_bin']} (exists: {item['bin_exists']})")
            print(f"  Expected idx: {item['expected_idx']} (exists: {item['idx_exists']})")
    else:
        print("\n✅ All input files have corresponding outputs!")


if __name__ == "__main__":
    main()
