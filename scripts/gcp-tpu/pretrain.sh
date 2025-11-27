MSC_CONFIG=configs/pretraining-examples/tpu/msc.yml \
    PJRT_DEVICE=TPU \
    TOKENIZERS_PARALLELISM=false \
    python -m lm_engine.pretrain --config ${1}
