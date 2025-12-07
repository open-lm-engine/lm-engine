INPUT_PATH=/data/tmp/test/part_000045.parquet
OUTPUT_PATH=/data/tmp/test-tokenized/part_000045
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8265 -- bash -c "rm -rf test && mkdir -p test && cd test && git clone --recurse-submodules https://github.com/open-lm-engine/lm-engine && source ../lm-engine/.venv/bin/activate && cd lm-engine && git checkout test && uv pip install "multi-storage-client[google-cloud-storage]" && MSC_CONFIG=/app/lm-engine/configs/pretraining-examples/tpu/msc.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 1 --download-locally"

# ray job stop --address  http://localhost:8265 03000000
