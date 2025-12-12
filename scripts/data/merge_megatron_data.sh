INPUT_PATH=/data/Nemotron-CC-v2/Medium-High-Quality
OUTPUT_PATH=/data/tmp2/Nemotron-CC-v2/Medium-High-Quality

ray job submit --address http://localhost:8266 -- bash -c "cd lm-engine && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python tools/data/merge_data.py --input-directory $INPUT_PATH --output-prefix $OUTPUT_PATH --max-size 250 --msc-base-path mayank-data --tmpdir /local-ssd --download-locally"