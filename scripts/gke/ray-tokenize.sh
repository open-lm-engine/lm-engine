INPUT_PATHS=(
"olmocr_science_pdfs-adult_content"
# "olmocr_science_pdfs-art_and_design"
# "olmocr_science_pdfs-crime_and_law"
# "olmocr_science_pdfs-education_and_jobs"
# "olmocr_science_pdfs-electronics_and_hardware"
# "olmocr_science_pdfs-entertainment"
# "olmocr_science_pdfs-fashion_and_beauty"
# "olmocr_science_pdfs-finance_and_business"
# "olmocr_science_pdfs-food_and_dining"
# "olmocr_science_pdfs-games"
# "olmocr_science_pdfs-health"
# "olmocr_science_pdfs-history_and_geography"
# "olmocr_science_pdfs-home_and_hobbies"
# "olmocr_science_pdfs-industrial"
# "olmocr_science_pdfs-literature"
# "olmocr_science_pdfs-politics"
# "olmocr_science_pdfs-religion"
# "olmocr_science_pdfs-science_math_and_technology-part2"
# "olmocr_science_pdfs-social_life"
# "olmocr_science_pdfs-software"
# "olmocr_science_pdfs-software_development"
# "olmocr_science_pdfs-sports_and_fitness"
# "olmocr_science_pdfs-transportation"
# "olmocr_science_pdfs-travel_and_tourism"
)


# INPUT_PATHS=(
# "finemath-3plus"
# "finemath-4plus"
# "infiwebmath-3plus"
# "infiwebmath-4plus"
# )

TOKENIZER=/app/lm-engine/tokenizers/granite-4.0-h-tiny-base

for ITEM in "${INPUT_PATHS[@]}"; do
    INPUT_PATH="/data/tmp3/dolma3-pool/data/$ITEM"
    OUTPUT_PATH="/data/tmp3/dolma3-pool-tokenized/data/$ITEM"
    
    ray job submit --address http://localhost:8265 -- bash -c "cd lm-engine && git fetch && git reset --hard origin/test && uv pip install -e . && MSC_CONFIG=/app/lm-engine/configs/msc/gcs.yml python tools/data/preprocess_data.py --input $INPUT_PATH --tokenizer $TOKENIZER --output-prefix $OUTPUT_PATH --append-eod --ray-workers 20000 --download-locally --msc-base-path mayank-data --tmpdir /local-ssd"
done

# ray job stop --address  http://localhost:8265 03000000
