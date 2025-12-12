#!/usr/bin/env python3
"""
Script to download specific folders from allenai/dolma3_pool HuggingFace dataset
and upload them to Google Cloud Storage.

Usage:
    python download_dolma3_to_gcs.py [--dry-run] [--parallel WORKERS]

Requirements:
    pip install huggingface_hub google-cloud-storage
    
    You also need:
    - HuggingFace token with access to the dataset (set HF_TOKEN env var or login via `huggingface-cli login`)
    - Google Cloud credentials configured (via `gcloud auth application-default login` or service account)
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

# Configuration
HF_REPO_ID = "allenai/dolma3_pool"
GCS_BUCKET = "mayank-data"
GCS_PREFIX = "tmp/dolma3-pool/data"

FOLDERS = [
    "olmocr_science_pdfs-adult_content",
    "olmocr_science_pdfs-art_and_design",
    "olmocr_science_pdfs-crime_and_law",
    "olmocr_science_pdfs-education_and_jobs",
    "olmocr_science_pdfs-electronics_and_hardware",
    "olmocr_science_pdfs-entertainment",
    "olmocr_science_pdfs-fashion_and_beauty",
    "olmocr_science_pdfs-finance_and_business",
    "olmocr_science_pdfs-food_and_dining",
    "olmocr_science_pdfs-games",
    "olmocr_science_pdfs-health",
    "olmocr_science_pdfs-history_and_geography",
    "olmocr_science_pdfs-home_and_hobbies",
    "olmocr_science_pdfs-industrial",
    "olmocr_science_pdfs-literature",
    "olmocr_science_pdfs-politics",
    "olmocr_science_pdfs-religion",
    "olmocr_science_pdfs-science_math_and_technology-part2",
    "olmocr_science_pdfs-social_life",
    "olmocr_science_pdfs-software",
    "olmocr_science_pdfs-software_development",
    "olmocr_science_pdfs-sports_and_fitness",
    "olmocr_science_pdfs-transportation",
    "olmocr_science_pdfs-travel_and_tourism",
]


def download_and_upload_folder(folder: str, cache_dir: str, dry_run: bool = False) -> dict:
    """Download a folder from HuggingFace and upload to GCS."""
    result = {"folder": folder, "status": "pending", "message": ""}
    
    try:
        print(f"[{folder}] Starting download (this may take a while)...")
        
        # Download the specific folder from HuggingFace
        local_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"data/{folder}/*"],
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"[{folder}] Download complete!")
        
        # Path to the downloaded folder
        source_path = Path(local_path) / "data" / folder
        
        if not source_path.exists():
            result["status"] = "error"
            result["message"] = f"Downloaded path does not exist: {source_path}"
            return result
        
        # GCS destination
        gcs_dest = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{folder}/"
        
        print(f"[{folder}] Uploading to {gcs_dest}...")
        
        if dry_run:
            print(f"[{folder}] DRY RUN: Would upload {source_path} to {gcs_dest}")
            result["status"] = "success (dry-run)"
            result["message"] = f"Would upload to {gcs_dest}"
        else:
            # Use gsutil for efficient parallel upload
            cmd = [
                "gsutil", "-m", "cp", "-r",
                str(source_path) + "/*",
                gcs_dest
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            result["status"] = "success"
            result["message"] = f"Uploaded to {gcs_dest}"
            print(f"[{folder}] Successfully uploaded to {gcs_dest}")
        
    except subprocess.CalledProcessError as e:
        result["status"] = "error"
        result["message"] = f"gsutil error: {e.stderr}"
        print(f"[{folder}] ERROR: {e.stderr}")
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        print(f"[{folder}] ERROR: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download dolma3_pool folders from HuggingFace and upload to GCS"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually uploading",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel downloads/uploads (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloads (default: temp directory)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Specific folders to process (default: all folders)",
    )
    args = parser.parse_args()
    
    folders_to_process = args.folders if args.folders else FOLDERS
    
    print(f"Processing {len(folders_to_process)} folders...")
    print(f"Source: {HF_REPO_ID}")
    print(f"Destination: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"Parallel workers: {args.parallel}")
    if args.dry_run:
        print("DRY RUN MODE - no actual uploads will occur")
    print("-" * 60)
    
    # Use provided cache dir or create temp directory
    if args.cache_dir:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        cleanup_cache = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="dolma3_download_")
        cache_dir = temp_dir
        cleanup_cache = True
    
    print(f"Using cache directory: {cache_dir}")
    
    results = []
    
    try:
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        download_and_upload_folder, folder, cache_dir, args.dry_run
                    ): folder
                    for folder in folders_to_process
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            for folder in folders_to_process:
                result = download_and_upload_folder(folder, cache_dir, args.dry_run)
                results.append(result)
    
    finally:
        if cleanup_cache:
            print(f"\nCleaning up temp directory: {cache_dir}")
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["status"].startswith("success")]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['folder']}: {r['message']}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())

