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


class Mode(Enum):
    """training / inference mode"""

    training = "training"
    inference = "inference"
    unsharding = "unsharding"
    distillation = "distillation"


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
    # cute kernels
    causal_conv1d = "causal_conv1d"
    continuous_count_cute = "continuous_count_cute"
    cross_entropy_cute = "cross_entropy_cute"
    fused_linear_cross_entropy_cute = "fused_linear_cross_entropy_cute"
    gru_cute = "gru_cute"
    pack_sequence_cute = "pack_sequence_cute"
    rmsnorm_cute = "rmsnorm_cute"
    rmsnorm_memory_efficient_cute = "rmsnorm_memory_efficient_cute"
    rnn_cute = "rnn_cute"
    swiglu_packed_cute = "swiglu_packed_cute"
    unpack_sequence_cute = "unpack_sequence_cute"
    # external kernels
    flash_attention_2 = "flash_attention_2"
    flash_attention_3 = "flash_attention_3"
    mamba2_ssm = "mamba2_ssm"
    scattermoe = "scattermoe"
    # custom functions
    ladder_residual_overlapped_layer = "ladder_residual_overlapped_layer"
