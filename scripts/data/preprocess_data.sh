INPUT_PATH=data.jsonl
OUTPUT_PATH=data
TOKENIZER=ibm-granite/granite-3b-code-base

python tools/data/preprocess_data.py \
    --input $INPUT_PATH \
    --tokenizer $TOKENIZER \
    --output-prefix $OUTPUT_PATH \
    --append-eod
