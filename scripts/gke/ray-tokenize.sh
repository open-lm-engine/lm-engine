INPUT_PATH=data.jsonl
OUTPUT_PATH=data
TOKENIZER=ibm-granite/granite-3b-code-base

ray job submit --address http://localhost:8265 -- bash -c "source .venv/bin/activate && cd lm-engine && python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod"
