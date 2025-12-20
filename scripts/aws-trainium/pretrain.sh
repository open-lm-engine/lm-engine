TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --rdzv_id=101 \
    -m lm_engine.pretrain \
    --config ${1}
