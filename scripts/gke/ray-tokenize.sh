INPUT_PATH=/data/tmp/test/
OUTPUT_PATH=/data/tmp/test-tokenized/
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && source .venv/bin/activate && python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 1"
