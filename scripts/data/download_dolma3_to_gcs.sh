#!/bin/bash
# Script to download specific folders from allenai/dolma3_pool and upload to GCS
#
# Usage:
#   ./download_dolma3_to_gcs.sh [--dry-run]
#
# Prerequisites:
#   - huggingface-cli installed and logged in
#   - gcloud CLI configured with GCS access

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - no actual uploads will occur"
fi

HF_REPO="allenai/dolma3_pool"
GCS_BUCKET="gs://mayank-data/tmp3/dolma3-pool/data"
CACHE_DIR="${CACHE_DIR:-/tmp/dolma3_download}"

FOLDERS=(
    # "olmocr_science_pdfs-adult_content"
    # "olmocr_science_pdfs-art_and_design"
    # "olmocr_science_pdfs-crime_and_law"
    # "olmocr_science_pdfs-education_and_jobs"
    # "olmocr_science_pdfs-electronics_and_hardware"
    # "olmocr_science_pdfs-entertainment"
    # "olmocr_science_pdfs-fashion_and_beauty"
    # "olmocr_science_pdfs-finance_and_business"
    # "olmocr_science_pdfs-food_and_dining"
    # "olmocr_science_pdfs-games"
    # "olmocr_science_pdfs-health"
    # "olmocr_science_pdfs-history_and_geography"
    # "olmocr_science_pdfs-home_and_hobbies"
    # "olmocr_science_pdfs-industrial"
    # "olmocr_science_pdfs-literature"
    # "olmocr_science_pdfs-politics"
    # "olmocr_science_pdfs-religion"
    "olmocr_science_pdfs-science_math_and_technology-part2"
    "olmocr_science_pdfs-social_life"
    "olmocr_science_pdfs-software"
    "olmocr_science_pdfs-software_development"
    "olmocr_science_pdfs-sports_and_fitness"
    "olmocr_science_pdfs-transportation"
    "olmocr_science_pdfs-travel_and_tourism"
)

mkdir -p "$CACHE_DIR"

echo "========================================"
echo "Source: $HF_REPO"
echo "Destination: $GCS_BUCKET"
echo "Cache dir: $CACHE_DIR"
echo "Folders to process: ${#FOLDERS[@]}"
echo "========================================"

for folder in "${FOLDERS[@]}"; do
    echo ""
    echo ">>> Processing: $folder"
    
    LOCAL_PATH="$CACHE_DIR/$folder"
    GCS_PATH="$GCS_BUCKET/$folder/"
    
    # Download using huggingface-cli (handles large repos better)
    echo "    Downloading from HuggingFace..."
    hf download \
        --repo-type dataset \
        --include "data/$folder/*" \
        --local-dir "$CACHE_DIR" \
        "$HF_REPO"
    
    # Check if download succeeded
    if [[ ! -d "$CACHE_DIR/data/$folder" ]]; then
        echo "    ERROR: Download failed or folder empty"
        continue
    fi
    
    # Upload to GCS
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    DRY RUN: Would upload $CACHE_DIR/data/$folder/* to $GCS_PATH"
    else
        echo "    Uploading to GCS..."
        gsutil -m cp -r "$CACHE_DIR/data/$folder"/* "$GCS_PATH"
        echo "    Done: $GCS_PATH"
    fi
    
    # Clean up to save disk space (optional - comment out to keep files)
    # rm -rf "$CACHE_DIR/data/$folder"
done

echo ""
echo "========================================"
echo "All folders processed!"
echo "========================================"

