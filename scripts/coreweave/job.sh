#!/bin/bash
#SBATCH --partition=gb200
#SBATCH --nodes=16
#SBATCH --job-name=lm-engine-job
#SBATCH --ntasks-per-node=1  #<--must be 1 for torchrun / override for others like mpi
#SBATCH --gpus-per-node=4
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0
#SBATCH --segment=16 # 9 18 <-- currently commented out and experimenting how it impacts 256 node job

####SBATCH --exclusive <-- currently commented out and experimenting how it impacts 256 node job

#run this command on slurm login node: 
# sbatch -N 16 /mnt/home/bobcalio/ai-coreweave/lm_engine/scripts/cw-gb200/pretrain-120b.sbatch <config>

. $HOME/.bashrc
export HF_HOME="$HOME/.cache/huggingface/"
proj_dir="/mnt/vast/proj/checkpoints/$USER" #sets the place to write all logs 
min_bw=$( echo "140" | bc -l)

export WANDB_BASE_URL=https://api.wandb.ai
# export WANDB_PROJECT=SUT
# export WANDB_ENTITY=ete-dcgm-monitor
# export WANDB_NAME=120b-moe-test
export WANDB_DISABLE_CODE=1
export WANDB_DISABLE_GIT=1
export WANDB__SERVICE_WAIT=300
# export WANDB_RUN_ID=120b-256 #"${JOB_ID}" #wb-no-metrics
#move this down export WANDB_DIR=$WANDB_LOGS_PATH
#echo "WANDB_RUN_ID ${WANDB_RUN_ID}"
#export WANDB_MODE=offline #leverage wnadb for run stats, but do not upload to the cloud 

: "${PREFLIGHT_TEST:=0}"
: "${CLEANUP_TEMP_DIR:=0}"

config=$(realpath ${1})
echo $config
filename=$(basename $config)
expname=${filename%.*}

PYXIS_DEFAULTS=( '--no-container-mount-home' '--no-container-remap-root' '' )

container_name="cute-kernels"
container_image="/mnt/vast/squash/${container_name}.sqsh"
container_mounts="/mnt:/mnt"

# from MLPerf team -- need top review 
#. ${HOME}/ai-coreweave/lm_engine/scripts/cw-gb200/config_common.sh

#default nccl vars handled in .nccl.conf
export TOKENIZERS_PARALLELISM=false 
export NCCL_SOCKET_IFNAME=eth0
#export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export PMIX_MCA_gds='^ds12'
export NCCL_MIN_CTAS=32
export NCCL_NET_GDR_C2C=1
export NCCL_WORK_FIFO_DEPTH=1048576
##
export NCCL_TIMEOUT_WAIT_SEC=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch If set to 1, aborting NCCL communicator and tearing down process upon error.
export PYTHONUNBUFFERED=TRUE #<print right away 
export OMP_NUM_THREADS=64 #<--can adjust, but for now leave to remove warning 
#export OMP_DYNAMIC=true

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export JOB_ID=${SLURM_JOBID}

export NCCL_DEBUG=WARN #INFO 
export NCCL_DEBUG_SUBSYS=BOOTSTRAP,INIT,NET,ENV
export NCCL_DEBUG_FILE=${NCCL_LOGS_PATH}/NCCL_DEBUG_FILE.%h.txt

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch If set to 1, aborting NCCL communicator and tearing down process upon error.

export TORCH_CUDA_ARCH_LIST="Blackwell" # 12.0+PTX
export CUTE_ARCH_LDSM_SM100A_ENABLED=1 
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TRITON_HOME=/tmp/$USER/triton
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"
#save hostlist for replay / debug if needed 
echo $SLURM_JOB_NODELIST > $run_dir/hostfile-${SLURM_JOB_ID}.txt
mkdir -p $PWD/logs/
#setup some srun args 
SRUN_ARGS=" --kill-on-bad-exit=1  \
--container-image=${container_image}  \
--container-mounts=${container_mounts}  \
--no-container-remap-root \
--container-workdir=$PWD \
--output=$PWD/logs/out-$expname.log \
--error=$PWD/logs/err-$expname.log
"

# --container-workdir=/mnt/vast/proj/checkpoints/mayank/dolomite-engine \
echo $SRUN_ARGS

##############################################
# We should invoke srun below for preflight checks with exactly one of the following.
##############################################
# to run one task on rank 0: 'srun -N1 -n1'
# to run one task per node: 'srun --ntasks-per-node=1'
# to run one task per gpu: 'srun --ntasks-per-node=${GPUS_PER_NODE}'

#this next line just warms up the container on every node 

export DISTRIBUTED_ARGS="\
--nproc_per_node $GPUS_PER_NODE \
--node_rank $NODE_RANK \
--nnodes=$SLURM_JOB_NUM_NODES \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
"
echo $DISTRIBUTED_ARGS

export TRITON_PRINT_AUTOTUNING=1

# # srun ${SRUN_ARGS} pip install -e ../cute-kernels/
# command='bash -c "pip install -e ../cute-kernels/ && torchrun $DISTRIBUTED_ARGS  -m lm_engine.pretrain --config $config"'
# srun ${SRUN_ARGS} ${command}
# # Optional: install package before running
# # srun ${SRUN_ARGS} pip install -e ../cute-kernels/

command="torchrun $DISTRIBUTED_ARGS -m lm_engine.pretrain --config $config"
srun ${SRUN_ARGS} bash -c "$command"
