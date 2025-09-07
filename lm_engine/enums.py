# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from enum import Enum


class ParamsGroupMethod(Enum):
    mup = "mup"


class GradientCheckpointingMethod(Enum):
    block = "block"


class LRDecaySchedule(Enum):
    constant = "constant"
    cosine = "cosine"
    exponential = "exponential"
    linear = "linear"
    power = "power"


class DatasetSplit(Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class TuningMethod(Enum):
    """training method"""

    pretraining = "pretraining"
    full_finetuning = "full_finetuning"
    distillation = "distillation"


class LossMask(Enum):
    """Type of loss masking method"""

    output_only = "output_only"
    no_mask = "no_mask"


class KLDivergenceMethod(Enum):
    """Type of KL divergence"""

    forward = "forward"
    backward = "backward"


class ExperimentsTrackerName(Enum):
    """Experiment tracker to use"""

    aim = "aim"
    wandb = "wandb"


class Kernel(Enum):
    # FMA
    causal_conv1d = "causal_conv1d"
    continuous_count = "continuous_count"
    cross_entropy = "cross_entropy"
    fused_linear_cross_entropy = "fused_linear_cross_entropy"
    grouped_gemm = "grouped_gemm"
    gru = "gru"
    p_norm = "p_norm"
    pack_sequence = "pack_sequence"
    rmsnorm = "rmsnorm"
    rmsnorm_memory_efficient = "rmsnorm_memory_efficient"
    rnn = "rnn"
    swiglu_packed = "swiglu_packed"
    unpack_sequence = "unpack_sequence"
    # external kernels
    flash_attention_2 = "flash_attention_2"
    flash_attention_3 = "flash_attention_3"
    mamba2_ssm = "mamba2_ssm"
    scattermoe = "scattermoe"
    # custom functions
    checkpointed_mlp = "checkpointed_mlp"
    ladder_residual_overlapped_layer = "ladder_residual_overlapped_layer"
    # smoe
    smoe = "smoe"
