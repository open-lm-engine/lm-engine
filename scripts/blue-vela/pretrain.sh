export CUDA_DEVICE_ORDER=PCI_BUS_ID # Recommended 
export CUDA_MODULE_LOADING=LAZY # Recommended 
# export CUDA_DEVICE_MAX_CONNECTIONS=1 # Required 

# NCCl/InfiniBand options
export NCCL_IB_PCI_RELAXED_ORDERING=2 # only use if available and beneficial
export NCCL_IB_QPS_PER_CONNECTION=4 #<--April 16 8  #Feb 3 based on @rpands yaml
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_DISABLE=0
#comment the next two out - they were for debug 
# export NCCL_IB_TIMEOUT=16
# export NCCL_IB_RETRY_CNT=14 #<--April 16

#v--April 16 --v
#export NCCL_CHECKS_DISABLE=1 #should be better perf after Debug phase 
#export NCCL_CHECK_POINTERS=0 #should be better perf after Debug phase

#
#March 28 confirmed for DFW BlueVela same InfiniBand settings 
#exclude *storage* IB NICs  
export NCCL_IB_HCA="^=mlx5_1,mlx5_6"
#exclude storage IB NICs for Sockets too  
export NCCL_SOCKET_IFNAME="=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0"


#The NCCL_IGNORE_CPU_AFFINITY variable can be used to cause NCCL to ignore the job’s supplied CPU affinity and instead use the GPU affinity only.
# The default is 0, set to 1 to cause NCCL to ignore the job’s supplied CPU affinity.
#comment out April 16
#export NCCL_IGNORE_CPU_AFFINITY=1 #Feb 3 based on @rpands yaml
#export NCCL_CROSS_NIC=2 #Try to use the same NIC for the same ring/tree, but still allow for it if it would result in better performance.

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #for newer versions of pytorch 
#v--April 16
export NCCL_DEBUG_SUBSYS=NET,ENV,INIT #Feb 3 #INIT,COLL,P2P,SHM,NET,ENV #ALL
# export NCCL_DEBUG=INFO # Initial run to verify GDR after that set to ERROR  ##TRACE #INFO WARN

#comment out April 16
#export NCCL_SOCKET_NTHREADS=4 #not really needed
#export NCCL_NSOCKS_PERTHREAD=4 #not really needed
#export NCCL_BUFFSIZE=8388608 # 8MB seems to work on H100
#export NCCL_MIN_NCHANNELS=32 #H100s 

export OMP_NUM_THREADS=64 #<--April 16    

#Enable the use of NVLink SHARP (NVLS). NVLink SHARP is available in third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures, 
#allowing collectives such as ncclAllReduce to be offloaded to the NVSwitch domain.
#comment out April 16
export NCCL_NVLS_ENABLE=1  #1(on) is the default anyway if available 

#export NCCL_NET_GDR_LEVEL=0
#export NCCL_SHM_DISABLE=1


MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=28444 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(echo $HOSTNAME | cut -d'.' -f1) | cut -d':' -f1)-1))

#track the LSF/Slurm job in W&B per run - bobcalio
export JOB_ID=$LSB_JOBID 

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m lm_engine.pretrain \
    --config ${1}
