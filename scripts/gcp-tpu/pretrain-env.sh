# MSC_CONFIG=configs/msc/gcs.yml \
#     PJRT_DEVICE=TPU \
#     TOKENIZERS_PARALLELISM=false \
#     python -m lm_engine.pretrain --config ${1}
# Format: [weight, "path/to/data/prefix", weight, "path/to/data/prefix"]
export DATA_MIXTURE='[
    1.0, "gs://mayank-data/finemath/finemath-4plus/0"
]'
export MSC_CONFIG='/app/lm-engine/configs/msc/gcs.yml'


export COMMAND="PJRT_DEVICE=TPU \
    TOKENIZERS_PARALLELISM=false \
    python -m lm_engine.pretrain configs/pretraining-examples/tpu/pretrain.yml"

envsubst < scripts/gke/tpu/tpu-1.yml | kubectl apply -f -

echo "Job submitted. Waiting for pod to initialize..."
