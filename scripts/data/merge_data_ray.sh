INPUT_PATH="/data/tmp-tokenized/stack-edu"
OUTPUT_PATH="/data/stack-edu"

ray job submit --address http://localhost:8270 -- bash -c "cd lm-engine && git fetch && git reset --hard origin/test && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python tools/data/merge_data.py --input-directory $INPUT_PATH --output-prefix $OUTPUT_PATH --max-size 250 --msc-base-path mayank-data --tmpdir /local-ssd --download-locally --workers 10"

# && git fetch && git reset --hard origin/test && uv pip install -e . 
