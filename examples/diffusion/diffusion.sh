#!/bin/bash
set -x
DATASET="Zyphra/dclm-dedup"
BASE_TOKENIZER="openai-community/gpt2"
DATA_PATH="../data/"
mkdir -p $DATA_PATH
TRAIN_PATH="$DATA_PATH/dclm-dedup-gpt2-tokenized"
mkdir -p $TRAIN_PATH
TOKENIZER_PATH="$DATA_PATH/tokenizer"
mkdir -p $TOKENIZER_PATH

python -u examples/diffusion/modify_tokenizer.py --tokenizer $BASE_TOKENIZER --output-path $TOKENIZER_PATH

CHUNK=0
CHUNK_SIZE=20000000
START_IDX=$(($CHUNK * $CHUNK_SIZE))
END_IDX=$(($START_IDX + $CHUNK_SIZE))
SPLIT="train[$START_IDX:$END_IDX]"

OUTPUT_FILE="$TRAIN_PATH/dclm_`printf '%02d' $CHUNK`"
python -u examples/diffusion/preprocess_data.py \
	--input Zyphra/dclm-dedup --split $SPLIT \
	--tokenizer $TOKENIZER_PATH \
	--output-prefix $OUTPUT_FILE \
	--workers 128 --chunk-size 8192 --append-eod
