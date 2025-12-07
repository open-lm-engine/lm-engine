INPUT_PATH=/data/tmp/Nemotron-CC-v2/Translated-Diverse-QA/
OUTPUT_PATH=/data/tmp/Nemotron-CC-v2-tokenized/Translated-Diverse-QA/
TOKENIZER=ibm-granite/granite-4.0-h-tiny-base

ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && source .venv/bin/activate && python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 1"
