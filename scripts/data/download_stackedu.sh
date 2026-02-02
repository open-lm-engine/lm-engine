INPUT_PATH="/data/tmp/the-stack-v2-dedup"
OUTPUT_PATH="/data/tmp-hans/the-stack-v2-dedup-expanded"

ray job submit --address http://localhost:8270 -- bash -c "cd lm-engine && git fetch && git reset --hard origin/test && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python scripts/data/download_stackedu.py --input-directory $INPUT_PATH --output-prefix $OUTPUT_PATH  --msc-base-path mayank-data --tmpdir /local-ssd --max-concurrent 1000 --download-locally --secret-access-key $SECRET_ACCESS_KEY --access-key-id $ACCESS_KEY_ID"
