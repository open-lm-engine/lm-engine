MSC_CONFIG=configs/msc/gcs.yml \
    PJRT_DEVICE=TPU \
    TOKENIZERS_PARALLELISM=false \
    python -m lm_engine.pretrain --config ${1}
