export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDRESS=$(hostname) # use IP addr of headnode for communication
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m lm_engine.pretrain \
    --config ${1}
