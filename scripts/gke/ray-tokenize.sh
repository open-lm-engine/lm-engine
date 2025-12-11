# INPUT_PATH=/data/tmp/dolma3-pool
INPUT_PATH=/data/tmp/dolma3-pool/data/common_crawl-education_and_jobs-0014/shard_00000000.jsonl.zst
OUTPUT_PATH=/data/tmp/dolma3-pool-tokenized
# INPUT_PATH=/data/tmp/test
# OUTPUT_PATH=/data/tmp/test-tokenized
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 2000 --download-locally --msc-base-path mayank-data --tmpdir /local-ssd"

# ray job stop --address  http://localhost:8265 03000000