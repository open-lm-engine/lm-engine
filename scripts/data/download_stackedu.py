# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""
Download files from The Stack v2 dataset using Ray for parallelization
and aiobotocore for async parallel S3 downloads.

Each parquet file in the input directory contains blob IDs (one per row).
This script submits a Ray job per file to download all blobs in parallel.
"""
import argparse
import asyncio
import gzip
import logging
import os
import tempfile
import traceback

import aiobotocore.session
import multistorageclient as msc
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from tqdm import tqdm

from lm_engine.defaults import MSC_PREFIX


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# S3 bucket for The Stack v2 (SWH - Software Heritage)
# URL format: s3://softwareheritage/content/{blob_id}
STACK_S3_BUCKET = "softwareheritage"
STACK_S3_PREFIX = "content"


def _convert_path_to_msc_path(path: str, base_msc_path: str) -> str:
    """Convert local path to MSC path."""
    path = path.lstrip(os.sep)
    # Split the first component (assumed to be mount point name like 'data')
    _, base_path = path.split(os.sep, 1)
    msc_path = os.path.join(base_msc_path, base_path)
    return f"{MSC_PREFIX}{msc_path}"


async def download_blob_from_s3(
    s3_client,
    blob_id: str,
    src_encoding: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Download a single blob from S3 using async aiobotocore.

    The Stack v2 stores blobs at: s3://softwareheritage/content/{blob_id}
    Content is gzip compressed.
    """
    async with semaphore:
        try:
            # S3 key format: content/{blob_id}
            s3_key = f"{STACK_S3_PREFIX}/{blob_id}"

            # Download using aiobotocore
            response = await s3_client.get_object(Bucket=STACK_S3_BUCKET, Key=s3_key)

            # Read and decompress gzip content
            async with response["Body"] as stream:
                compressed_data = await stream.read()
                # Decompress gzip content
                content = gzip.decompress(compressed_data).decode(src_encoding)

            return {"status": "success", "blob_id": blob_id, "content": content}
        except Exception as e:
            return {"status": "error", "blob_id": blob_id, "error": str(e), "content": None}


async def _download_all_blobs(
    blob_ids: list[str],
    src_encodings: list[str],
    max_concurrent: int,
    secret_access_key: str,
    access_key_id: str,
    source_basename: str,
) -> tuple[dict[str, str], int, int]:
    """
    Download all blobs using async aiobotocore with semaphore for concurrency control.
    Returns (blob_id_to_content dict, success_count, error_count).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    success_count = 0
    error_count = 0
    blob_id_to_content = {}

    session = aiobotocore.session.get_session()

    async with session.create_client(
        "s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key
    ) as s3_client:
        # Create all download tasks
        tasks = []
        for blob_id, src_encoding in zip(blob_ids, src_encodings):
            if blob_id is None:
                continue
            blob_id_str = str(blob_id)
            encoding = src_encoding if src_encoding else "utf-8"
            task = download_blob_from_s3(s3_client, blob_id_str, encoding, semaphore)
            tasks.append(task)

        # Execute all tasks with progress bar
        with tqdm(total=len(tasks), desc=f"Downloading {source_basename}", unit="blob") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result["status"] == "success":
                    success_count += 1
                    blob_id_to_content[result["blob_id"]] = result["content"]
                else:
                    error_count += 1
                    logger.warning(f"Failed to download {result['blob_id']}: {result.get('error')}")
                pbar.update(1)

    return blob_id_to_content, success_count, error_count


@ray.remote
def process_parquet_file(
    source_path: str,
    output_prefix: str,
    msc_base_path: str,
    tmpdir: str,
    max_concurrent: int,
    download_locally: bool,
    secret_access_key: str,
    access_key_id: str,
):
    """
    Process a single parquet file containing blob IDs.
    Downloads all blobs listed in the file using async parallel S3 downloads.
    """
    # Force configure logging on the worker
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    try:
        # Create temp directory for downloads
        os.makedirs(tmpdir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=tmpdir) as local_tmpdir:
            # Download the parquet file from MSC if needed
            if download_locally:
                msc_source = _convert_path_to_msc_path(source_path, msc_base_path)
                local_parquet = os.path.join(local_tmpdir, "input.parquet")
                msc.download_file(msc_source, local_parquet)
                parquet_path = local_parquet
            else:
                parquet_path = source_path

            # Read the parquet file to get blob IDs and encodings
            table = pq.read_table(parquet_path)

            # The Stack v2 uses 'blob_id' column for the blob identifier
            if "blob_id" not in table.column_names:
                raise ValueError(
                    f"Could not find 'blob_id' column in {source_path}. " f"Available columns: {table.column_names}"
                )

            blob_ids = table["blob_id"].to_pylist()

            # Get source encodings (used for decoding the gzip content)
            if "src_encoding" in table.column_names:
                src_encodings = table["src_encoding"].to_pylist()
            else:
                src_encodings = ["utf-8"] * len(blob_ids)

            logger.info(f"[{os.path.basename(source_path)}] Found {len(blob_ids)} blobs to download")

            # Download blobs in parallel using async aiobotocore
            blob_id_to_content, success_count, error_count = asyncio.run(
                _download_all_blobs(
                    blob_ids=blob_ids,
                    src_encodings=src_encodings,
                    max_concurrent=max_concurrent,
                    secret_access_key=secret_access_key,
                    access_key_id=access_key_id,
                    source_basename=os.path.basename(source_path),
                )
            )

            # Create output table with just blob_id and content columns
            content_column = [blob_id_to_content.get(blob_id) for blob_id in blob_ids]
            output_table = pa.table({"blob_id": blob_ids, "content": content_column})

            # Determine output path (same structure as input)
            rel_path = os.path.relpath(source_path, "/data")
            output_file = os.path.join(output_prefix, rel_path)
            local_output_file = os.path.join(local_tmpdir, "output.parquet")

            # Write to local parquet file
            pq.write_table(output_table, local_output_file, compression="zstd")
            logger.info(f"[{os.path.basename(source_path)}] Wrote {len(output_table)} rows to parquet")

            # Upload to MSC if needed
            if download_locally:
                msc_target = _convert_path_to_msc_path(output_file, msc_base_path)
                msc.upload_file(msc_target, local_output_file)
                logger.info(f"[{os.path.basename(source_path)}] Uploaded to {msc_target}")
            else:
                # Write directly to output path
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                pq.write_table(output_table, output_file, compression="zstd")

            return {
                "status": "success",
                "source": source_path,
                "total_blobs": len(blob_ids),
                "success_count": success_count,
                "error_count": error_count,
            }

    except Exception as e:
        return {
            "status": "error",
            "source": source_path,
            "error": str(e) + "\n" + traceback.format_exc(),
        }


def list_parquet_files(input_dir: str) -> list[str]:
    """List all parquet files recursively."""
    logger.info(f"Listing files in {input_dir}")

    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".parquet"):
                files.append(os.path.join(root, filename))

    logger.info(f"Found {len(files)} parquet files")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Download The Stack v2 blobs using Ray and multithreaded S3 downloads"
    )
    parser.add_argument(
        "--input-directory",
        type=str,
        required=True,
        help="Input directory containing parquet files with blob IDs",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix for downloaded blobs",
    )
    parser.add_argument(
        "--msc-base-path",
        type=str,
        required=True,
        help="MSC base path (bucket) to use for download/upload",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        default="/tmp",
        help="Temporary directory for local downloads (default: /tmp)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent blob downloads per file (default: 100)",
    )
    parser.add_argument(
        "--download-locally",
        action="store_true",
        help="Download parquet files locally before processing",
    )
    parser.add_argument(
        "--secret-access-key",
        type=str,
        required=True,
        help="AWS secret access key",
    )
    parser.add_argument(
        "--access-key-id",
        type=str,
        required=True,
        help="AWS access key ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files without processing",
    )

    args = parser.parse_args()

    logger.info(f"Input directory: {args.input_directory}")
    logger.info(f"Output prefix: {args.output_prefix}")
    logger.info(f"MSC base path: {args.msc_base_path}")
    logger.info(f"Temp directory: {args.tmpdir}")
    logger.info(f"Max concurrent downloads: {args.max_concurrent}")

    # List all parquet files
    parquet_files = list_parquet_files(args.input_directory)

    if args.dry_run:
        logger.info("[DRY RUN] Would process the following files:")
        for f in parquet_files:
            logger.info(f"  - {f}")
        return

    if not parquet_files:
        logger.info("No parquet files found to process")
        return

    # Initialize Ray
    ray.init(
        address="auto",
        runtime_env={"env_vars": {"MSC_CONFIG": os.environ.get("MSC_CONFIG", "")}},
        log_to_driver=True,
    )
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    # Submit Ray jobs for each parquet file
    futures = []
    for parquet_file in parquet_files:
        future = process_parquet_file.remote(
            source_path=parquet_file,
            output_prefix=args.output_prefix,
            msc_base_path=args.msc_base_path,
            tmpdir=args.tmpdir,
            max_concurrent=args.max_concurrent,
            download_locally=args.download_locally,
            secret_access_key=args.secret_access_key,
            access_key_id=args.access_key_id,
        )
        futures.append(future)

    logger.info(f"Submitted {len(futures)} Ray jobs")

    # Wait for results with progress
    results = []
    with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            for future in done:
                try:
                    result = ray.get(future)
                    results.append(result)
                    if result["status"] == "success":
                        logger.info(
                            f"✅ {os.path.basename(result['source'])}: "
                            f"{result['success_count']}/{result['total_blobs']} blobs downloaded"
                        )
                    else:
                        logger.error(f"✗ {result['source']}: {result['error']}")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                pbar.update(1)

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    total_blobs = sum(r.get("success_count", 0) for r in results if r["status"] == "success")

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {len(results)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {error_count}")
    logger.info(f"Total blobs downloaded: {total_blobs}")

    if error_count > 0:
        logger.info("\nFailed files:")
        for r in results:
            if r["status"] == "error":
                logger.info(f"  - {r['source']}: {r['error']}")


if __name__ == "__main__":
    main()
