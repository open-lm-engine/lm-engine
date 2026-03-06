export TRITON_PRINT_AUTOTUNING=1

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    -m lm_engine.pretrain \
    --config ${1}
