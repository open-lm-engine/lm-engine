# **************************************************
# Copyright (c) 2026, Mayank Mishra, Zhonglin Han
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
    # XMA
    causal_conv1d = "causal_conv1d"
    continuous_count = "continuous_count"
    cross_entropy = "cross_entropy"
    fused_linear_cross_entropy = "fused_linear_cross_entropy"
    gru = "gru"
    m2rnn = "m2rnn"
    pack_sequence = "pack_sequence"
    rmsnorm = "rmsnorm"
    rmsnorm_memory_efficient = "rmsnorm_memory_efficient"
    rnn = "rnn"
    scattermoe = "scattermoe"
    swiglu_packed = "swiglu_packed"
    unpack_sequence = "unpack_sequence"
    # quack
    quack_gemm = "quack_gemm"
    quack_gemm_act = "quack_gemm_act"
    quack_gemm_gated = "quack_gemm_gated"
    quack_rmsnorm = "quack_rmsnorm"
    # external kernels
    flash_attention_2 = "flash_attention_2"
    flash_attention_3 = "flash_attention_3"
    flash_attention_4 = "flash_attention_4"
    mamba2_ssm = "mamba2_ssm"
    sonicmoe = "sonicmoe"

    @classmethod
    def validate_enabled(cls, kernels: list["Kernel"]) -> None:
        enabled_xma_rmsnorm_kernels = [
            kernel for kernel in [cls.rmsnorm, cls.rmsnorm_memory_efficient] if kernel in kernels
        ]
        if cls.quack_rmsnorm in kernels and enabled_xma_rmsnorm_kernels:
            enabled_rmsnorm_kernels = [cls.quack_rmsnorm] + enabled_xma_rmsnorm_kernels
            kernel_names = ", ".join(kernel.value for kernel in enabled_rmsnorm_kernels)
            raise ValueError(f"quack_rmsnorm cannot be enabled with XMA RMSNorm kernels, got {kernel_names}")

        if cls.quack_gemm_gated in kernels and cls.swiglu_packed in kernels:
            raise ValueError("quack_gemm_gated cannot be enabled with swiglu_packed")

        if cls.quack_gemm_act in kernels and cls.quack_gemm_gated in kernels:
            raise ValueError("quack_gemm_act cannot be enabled with quack_gemm_gated")


class ContextParallelLoadBalancerMethod(Enum):
    headtail = "headtail"
