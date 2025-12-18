INPUT_PATH=/home/mayank/data/nemotron-cc-v2-tokenized/Diverse-QA
OUTPUT_PATH=/home/mayank/data/nemotron-cc-v2-merged/Diverse-QA

mkdir -p $OUTPUT_PATH

python tools/data/merge_data.py \
    --input-directory $INPUT_PATH \
    --output-prefix $OUTPUT_PATH \
    --max-size 250
