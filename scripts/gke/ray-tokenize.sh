INPUT_PATH="/data/tmp-hans/the-stack-v2-dedup-expanded/data"
OUTPUT_PATH="/data/tmp-tokenized/the-stack-v2-dedup"
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base
# If running on many nodes, use the local tokenizer so that huggingface doesn't complain about too many concurrent requests
# TOKENIZER=/app/lm-engine/tokenizers/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8270 -- bash -c "cd lm-engine && git fetch && git reset --hard origin/test && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --use-ray --download-locally --msc-base-path mayank-data --tmpdir /local-ssd"

# ray job stop --address  http://localhost:8265 03000000