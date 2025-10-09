#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --time={time_limit}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --account=laionize
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/%x_%j.out
#SBATCH --exclude=jwb[0059,0067,0069,0193,0198,0215,0266,0284,0287,0294,0359,0392,0418,0637,0647,0829,0832,0838,0898,0907,0921,0971,1004,1023,1029,1213,0760,0676,0096]

export SCRATCH_TMP=/p/scratch/cjsc
export APPTAINER_CACHEDIR=$SCRATCH_TMP/container_tmp/APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=$SCRATCH_TMP/container_tmp/APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE="/p/scratch/ccstdl/marianna/pytorch_24.09-py3.sif"


# Training setup
GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR="${MASTER_ADDR}i"

MASTER_PORT=12345
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

export GLOO_SOCKET_IFNAME=ib0 # for GLOO

# Data args
DATA=/p/data1/mmlaion/cherti1/megatron_lm
DATA_PATH="$DATA/starcoder_c_tokenized/preprocessed_content_document"
VOCAB_FILE="$DATA/vocab.json"
MERGE_FILE="$DATA/merges.txt"

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)


NUM_LAYERS={num_layers}
HIDDEN_SIZE={hidden_size}
NUM_ATTN_HEADS={num_attn_heads}
MAX_POSITION_EMBEDDINGS={seq_len}
SEQ_LENGTH={seq_len}
PP={pipeline_parallel}
TP={tensor_parallel}
GBS={global_batch_size}
MBS={micro_batch_size}


GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS 
    --hidden-size $HIDDEN_SIZE 
    --num-attention-heads $NUM_ATTN_HEADS 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
)

# Training args
TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters 100
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 43
    --use-distributed-optimizer
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
    --sequence-parallel
)

CHECKPOINT_PATH="/p/scratch/laionize/marianna/megatron/checkpoints"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
CHECKPOINT_PATH="$CHECKPOINT_PATH/$TIMESTAMP-gpt2-7b"

mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$CHECKPOINT_PATH/tensorboard"
mkdir -p $TENSORBOARD_LOGS_PATH


# Eval and logging args
EVAL_AND_LOGGING_ARGS=(
    --log-interval 5
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --log-throughput
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Command
CMD="pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    "

LAUNCHER="singularity \
    exec \
    --bind /p/scratch/laionize/marianna/megatron:/p/scratch/laionize/marianna/megatron \
    --bind /p/data1/mmlaion/cherti1/megatron_lm:/p/data1/mmlaion/cherti1/megatron_lm \
    --nv \
    $IMAGE \
    python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend static \
    --max_restarts 0 \
    --tee 3 \
    "
echo $CMD


SRUN_ARGS=" \
    --wait=60 --cpus-per-task=48 --threads-per-core=1 \
    --kill-on-bad-exit=1 \
    "

MEGATRON_PATH="/p/project1/laionize/marianna/megatron/Megatron-LM"
cd $MEGATRON_PATH
srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"
