# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

#!/usr/bin/env python3
import argparse
import logging
import os
import tempfile
import traceback

import multistorageclient as msc
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from google.cloud import storage
from tqdm import tqdm

from lm_engine.defaults import MSC_PREFIX


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants - paths on mounted filesystems
GCS_MOUNT = "/data"
INPUT_PATH = "/data/tmp-tokenized/the-stack-v2-dedup"
OUTPUT_PATH = "/data/tmp-hans/the-stack-v2-dedup-expanded/data"
DEFAULT_THRESHOLD_GB = 5
DEFAULT_TARGET_MB = 2048  # 1GB in MB
BYTES_PER_GB = 1024**3
BYTES_PER_MB = 1024**2
LOCAL_SSD_PATH = "/local-ssd-fast"


def list_parquet_files(input_dir: str) -> list[dict]:
    """List all parquet files recursively with their sizes."""
    logger.info(f"Listing files in {input_dir}")

    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".parquet"):
                filepath = os.path.join(root, filename)
                try:
                    size = os.path.getsize(filepath)
                    files.append(
                        {
                            "path": filepath,
                            "size": size,
                        }
                    )
                except OSError as e:
                    logger.warning(f"Could not get size of {filepath}: {e}")

    logger.info(f"Found {len(files)} parquet files")
    return files


def _delete_gcs_file(msc_path: str) -> None:
    """Delete a file from GCS given an MSC path (msc://bucket/path)."""
    # Strip the msc:// prefix
    path_without_prefix = msc_path[len(MSC_PREFIX) :]
    # Split into bucket and blob path
    bucket_name, blob_name = path_without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def _convert_path_to_msc_path_and_tmp_path(path: str, base_msc_path: str, tmpdir: str) -> tuple[str, str]:
    """Convert path to MSC path and local temp path.
    Assumes path starts with /data (GCS mount) or similar mount point.
    """
    path = path.lstrip(os.sep)
    # Split the first component (assumed to be mount point name like 'data')
    _, base_path = path.split(os.sep, 1)

    msc_path = os.path.join(base_msc_path, base_path)
    msc_path = f"{MSC_PREFIX}{msc_path}"

    local_path = os.path.join(tmpdir, base_path)
    return msc_path, local_path


@ray.remote
def process_single_file(source_path, input_base, output_base, target_size_mb, msc_base_path):
    """
    Download/Copy file to local SSD, split it, and upload/copy back.
    """
    # Force configure logging on the worker
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

    try:
        # Create temp directory on local SSD
        os.makedirs(LOCAL_SSD_PATH, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_SSD_PATH) as tmpdir:

            # 1. Download from MSC
            # We assume msc_base_path is provided as we are operating on /data paths
            msc_source_path, local_input_file = _convert_path_to_msc_path_and_tmp_path(
                source_path, msc_base_path, tmpdir
            )

            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_input_file), exist_ok=True)

            msc.download_file(msc_source_path, local_input_file)
            logger.info(f"[{os.path.basename(source_path)}] Downloaded to {local_input_file}")

            # 2. Split and Upload
            uploaded_count = 0

            # Calculate output directory structure relative to input_base
            rel_dir = os.path.dirname(source_path)
            if rel_dir.startswith(input_base):
                rel_dir = rel_dir[len(input_base) :].lstrip(os.sep)

            # Determine target directory for uploads
            target_dir = os.path.join(output_base, rel_dir)

            # Determine base filename for chunks
            filename = os.path.basename(local_input_file)
            target_file_base = os.path.splitext(filename)[0]

            with pq.ParquetFile(local_input_file) as pf:
                schema = pf.schema_arrow

                current_chunk = []
                current_size = 0
                chunk_index = 0

                logger.info(
                    f"[{os.path.basename(source_path)}] Found {pf.num_row_groups} row groups in {local_input_file}"
                )

                for i in tqdm(range(pf.num_row_groups), desc="Reading row groups", unit="group"):
                    rg = pf.read_row_group(i)

                    for batch in rg.to_batches():
                        current_chunk.append(batch)
                        current_size += batch.nbytes

                        if current_size >= target_size_mb * BYTES_PER_MB:
                            chunk_filename = f"{target_file_base}_{chunk_index:05d}.parquet"
                            chunk_local_path = os.path.join(tmpdir, chunk_filename)

                            table = pa.Table.from_batches(current_chunk, schema=schema)
                            pq.write_table(table, chunk_local_path)

                            # Upload immediately
                            target_path = os.path.join(target_dir, chunk_filename)
                            msc_target_path, _ = _convert_path_to_msc_path_and_tmp_path(
                                target_path, msc_base_path, tmpdir
                            )
                            msc.upload_file(msc_target_path, chunk_local_path)

                            # Delete local copy
                            os.remove(chunk_local_path)

                            uploaded_count += 1
                            current_chunk = []
                            current_size = 0
                            chunk_index += 1

                # Last chunk
                if current_chunk:
                    chunk_filename = f"{target_file_base}_{chunk_index:05d}.parquet"
                    chunk_local_path = os.path.join(tmpdir, chunk_filename)

                    table = pa.Table.from_batches(current_chunk, schema=schema)
                    pq.write_table(table, chunk_local_path)

                    # Upload immediately
                    target_path = os.path.join(target_dir, chunk_filename)
                    msc_target_path, _ = _convert_path_to_msc_path_and_tmp_path(target_path, msc_base_path, tmpdir)
                    msc.upload_file(msc_target_path, chunk_local_path)

                    # Delete local copy
                    os.remove(chunk_local_path)

                    uploaded_count += 1

            # 4. Delete original file from GCS bucket
            _delete_gcs_file(msc_source_path)
            logger.info(f"[{os.path.basename(source_path)}] Deleted original file from GCS: {msc_source_path}")

            return {"status": "success", "source": source_path, "chunks": uploaded_count}

    except Exception as e:
        return {
            "status": "error",
            "source": source_path,
            "error": str(e) + "\n" + traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="Split large parquet files into smaller chunks using Ray")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without actually splitting files",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=INPUT_PATH,
        help=f"Input directory on mounted GCS (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH,
        help=f"Output directory on mounted GCS (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--msc-base-path",
        type=str,
        required=True,
        help="MSC base path (bucket) to use for download/upload.",
    )

    args = parser.parse_args()

    threshold_bytes = int(DEFAULT_THRESHOLD_GB * BYTES_PER_GB)

    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"MSC Base Path: {args.msc_base_path}")

    # List all parquet files
    files = list_parquet_files(args.input_path)

    # Filter to files larger than threshold
    large_files = [f for f in files if f["size"] > threshold_bytes]
    print(large_files)

    logger.info(f"Found {len(large_files)} files larger than {DEFAULT_THRESHOLD_GB} GB")

    for f in large_files:
        size_gb = f["size"] / BYTES_PER_GB
        logger.info(f"  - {f['path']} ({size_gb:.2f} GB)")

    if args.dry_run:
        logger.info("[DRY RUN] Would process the above files")
        return

    if not large_files:
        logger.info("No files to process")
        return

    ray.init(
        address="auto",
        runtime_env={"env_vars": {"MSC_CONFIG": os.environ.get("MSC_CONFIG", "")}},
        log_to_driver=True,
    )
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    # Submit all tasks
    futures = []
    for file_info in large_files:
        future = process_single_file.remote(
            source_path=file_info["path"],
            input_base=args.input_path,
            output_base=args.output_path,
            target_size_mb=DEFAULT_TARGET_MB,
            msc_base_path=args.msc_base_path,
        )
        futures.append(future)

    logger.info(f"Submitted {len(futures)} tasks to Ray")

    # Wait for results with progress bar
    results = []

    with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            for future in done:
                try:
                    result = ray.get(future)
                    results.append(result)
                    if result["status"] == "success":
                        logger.info(f"✅ {os.path.basename(result['source'])} -> {result['chunks']} chunks")
                    else:
                        logger.error(f"✗ {result['source']}: {result['error']}")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                pbar.update(1)

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    total_chunks = sum(r["chunks"] for r in results if r["status"] == "success")

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {len(results)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {error_count}")
    logger.info(f"Total chunks created: {total_chunks}")

    if error_count > 0:
        logger.info(f"\nFailed files:")
        for r in results:
            if r["status"] == "error":
                logger.info(f"  - {r['source']}: {r['error']}")


if __name__ == "__main__":
    main()
