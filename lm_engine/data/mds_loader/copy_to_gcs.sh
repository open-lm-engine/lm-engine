#!/bin/bash

# Copy all files from ~/data/lhc/mds to gs://nemocc/

SOURCE_DIR=~/data/lhc/mds
DEST_BUCKET=gs://nemocc

echo "Copying files from $SOURCE_DIR to $DEST_BUCKET (using up to 6 parallel threads)"

# Use find and xargs to copy files in parallel (up to 6 threads)
find "$SOURCE_DIR" -maxdepth 1 -type f -print0 | \
    xargs -0 -P 6 -I {} sh -c '
        filename=$(basename "{}")
        echo "Copying $filename..."
        gsutil cp "{}" "'"$DEST_BUCKET"'/$filename"
    '

echo "Done!"