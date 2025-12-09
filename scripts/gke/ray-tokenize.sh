INPUT_PATH=/data/tmp/test/part_000045.parquet
OUTPUT_PATH=/data/tmp/Nemotron-CC-v2-tokenized/test
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && MSC_CONFIG=/app/msc/gcs.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 16 --download-locally --msc-base-path mayank-data --tmpdir /local-ssd"

# ray job stop --address  http://localhost:8265 03000000
