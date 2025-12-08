INPUT_PATH=/data/tmp/test/part_000045.parquet
OUTPUT_PATH=/data/tmp/test-tokenized/part_000045
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

# ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && git pull && uv pip install -e . && python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 1 --download-locally --msc-config /app/msc/gcs.yml"

ray job submit --address http://localhost:8265 -- bash -c "rm -rf lm-engine && git clone --recurse-submodules https://github.com/open-lm-engine/lm-engine && cd lm-engine && git checkout test && uv pip install -e . && MSC_CONFIG=/app/msc/gcs.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 1 --download-locally"

# ray job stop --address  http://localhost:8265 03000000
