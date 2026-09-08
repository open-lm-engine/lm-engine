# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import triton
import triton.language as tl

from .....accelerator import Accelerator
from .....kernels.custom_op import xma_op
from .....math import ceil_divide, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_X in get_powers_of_2(4, 1024):
        for BLOCK_SIZE_Y in get_powers_of_2(4, 1024):
            for BLOCK_SIZE_Z in get_powers_of_2(4, 1024):
                total = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z
                if BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z > 8192:
                    continue

                for num_warps in get_powers_of_2(4, max(4, total >> 5)):
                    configs.append(
                        triton.Config(
                            {"BLOCK_SIZE_X": BLOCK_SIZE_X, "BLOCK_SIZE_Y": BLOCK_SIZE_Y, "BLOCK_SIZE_Z": BLOCK_SIZE_Z},
                            num_warps=num_warps,
                        )
                    )

    return configs


def _early_config_prune(configs: list[triton.Config], named_args: dict, **_kwargs) -> list[triton.Config]:
    Y = named_args.get("Y")
    Z = named_args.get("Z")

    pruned = []
    for config in configs:
        BLOCK_SIZE_Y = config.kwargs["BLOCK_SIZE_Y"]
        BLOCK_SIZE_Z = config.kwargs["BLOCK_SIZE_Z"]

        if Y is None:
            if BLOCK_SIZE_Y != 4 or BLOCK_SIZE_Z != 4:
                continue
        elif Z is None:
            if BLOCK_SIZE_Z != 4:
                continue

        pruned.append(config)

    return pruned


@triton.autotune(
    configs=_get_autotune_configs(),
    key=[],
    prune_configs_by={"early_config_prune": _early_config_prune},
    reset_to_zero=["u_norm_ptr"],
)
@triton.jit
def _single_tensor_hyperball_state_update_triton_kernel(
    exp_avg_ptr,
    exp_avg_stride,
    exp_avg_sq_ptr,
    exp_avg_sq_stride,
    dW_ptr,
    dW_stride,
    u_ptr,
    u_stride,
    u_norm_ptr,
    beta1,
    beta2,
    bc1,
    bc2,
    eps,
    X: tl.constexpr,
    Y: tl.constexpr,
    Z: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    NUM_TILES_X = tl.cdiv(X, BLOCK_SIZE_X)
    NUM_TILES = NUM_TILES_X

    if Y is not None:
        NUM_TILES_Y = tl.cdiv(Y, BLOCK_SIZE_Y)
        NUM_TILES *= NUM_TILES_Y

    if Z is not None:
        NUM_TILES_Z = tl.cdiv(Z, BLOCK_SIZE_Z)
        NUM_TILES *= NUM_TILES_Z

    BLOCK_ID = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)
    NUM_TILES_PER_BLOCK = tl.cdiv(NUM_TILES, NUM_BLOCKS)

    start = BLOCK_ID * NUM_TILES_PER_BLOCK
    end = min(start + NUM_TILES_PER_BLOCK, NUM_TILES)
    u_norm = 0.0

    for TILE_ID in range(start, end):
        BLOCK_ID_X = TILE_ID % NUM_TILES_X
        TILE_ID_REM = TILE_ID // NUM_TILES_X

        BLOCK_X = BLOCK_ID_X * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
        MASK = BLOCK_X[:, None, None] < X

        exp_avg_ptrs = exp_avg_ptr + BLOCK_X[:, None, None] * exp_avg_stride[0]
        exp_avg_sq_ptrs = exp_avg_sq_ptr + BLOCK_X[:, None, None] * exp_avg_sq_stride[0]
        dW_ptrs = dW_ptr + BLOCK_X[:, None, None] * dW_stride[0]
        u_ptrs = u_ptr + BLOCK_X[:, None, None] * u_stride[0]

        if Y is not None:
            BLOCK_ID_Y = TILE_ID_REM % NUM_TILES_Y
            TILE_ID_REM = TILE_ID_REM // NUM_TILES_Y

            BLOCK_Y = BLOCK_ID_Y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
            MASK = MASK & (BLOCK_Y[None, :, None] < Y)

            exp_avg_ptrs += BLOCK_Y[None, :, None] * exp_avg_stride[1]
            exp_avg_sq_ptrs += BLOCK_Y[None, :, None] * exp_avg_sq_stride[1]
            dW_ptrs += BLOCK_Y[None, :, None] * dW_stride[1]
            u_ptrs += BLOCK_Y[None, :, None] * u_stride[1]

        if Z is not None:
            BLOCK_ID_Z = TILE_ID_REM % NUM_TILES_Z

            BLOCK_Z = BLOCK_ID_Z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
            MASK = MASK & (BLOCK_Z[None, None, :] < Z)

            exp_avg_ptrs += BLOCK_Z[None, None, :] * exp_avg_stride[2]
            exp_avg_sq_ptrs += BLOCK_Z[None, None, :] * exp_avg_sq_stride[2]
            dW_ptrs += BLOCK_Z[None, None, :] * dW_stride[2]
            u_ptrs += BLOCK_Z[None, None, :] * u_stride[2]

        dW = tl.load(dW_ptrs, mask=MASK).to(tl.float32)

        exp_avg = tl.load(exp_avg_ptrs, mask=MASK).to(tl.float32)
        exp_avg *= beta1
        exp_avg += dW * (1 - beta1)
        tl.store(exp_avg_ptrs, exp_avg, mask=MASK)

        exp_avg_sq = tl.load(exp_avg_sq_ptrs, mask=MASK).to(tl.float32)
        exp_avg_sq *= beta2
        exp_avg_sq += dW * dW * (1 - beta2)
        tl.store(exp_avg_sq_ptrs, exp_avg_sq, mask=MASK)

        u = exp_avg * bc1 / (tl.sqrt(exp_avg_sq * bc2) + eps)
        tl.store(u_ptrs, u, mask=MASK)

        u_norm += tl.sum(u * u)

    tl.atomic_add(u_norm_ptr, u_norm, sem="relaxed")


def _get_persistent_grid(DIMS: int, sm_margin: int) -> Callable:
    def _num_tiles(kwargs: dict) -> int:
        num_tiles = ceil_divide(kwargs["X"], kwargs["BLOCK_SIZE_X"])

        if DIMS >= 2:
            num_tiles *= ceil_divide(kwargs["Y"], kwargs["BLOCK_SIZE_Y"])

        if DIMS == 3:
            num_tiles *= ceil_divide(kwargs["Z"], kwargs["BLOCK_SIZE_Z"])

        return num_tiles

    if DIMS not in (1, 2, 3):
        raise ValueError

    cores = Accelerator.get_core_count()

    return lambda kwargs: (min(cores - sm_margin, _num_tiles(kwargs)),)


@xma_op(mutates_args={"exp_avg", "exp_avg_sq", "u", "u_norm"})
def _single_tensor_hyperball_state_update_triton(
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    dW: torch.Tensor,
    u: torch.Tensor,
    u_norm: torch.Tensor,
    beta1: float,
    beta2: float,
    t: int,
    eps: float,
    sm_margin: int,
) -> None:
    DIMS = dW.dim()

    _single_tensor_hyperball_state_update_triton_kernel[_get_persistent_grid(DIMS, sm_margin)](
        exp_avg_ptr=exp_avg,
        exp_avg_stride=exp_avg.stride(),
        exp_avg_sq_ptr=exp_avg_sq,
        exp_avg_sq_stride=exp_avg_sq.stride(),
        dW_ptr=dW,
        dW_stride=dW.stride(),
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        beta1=beta1,
        beta2=beta2,
        bc1=1 / (1 - beta1**t),
        bc2=1 / (1 - beta2**t),
        eps=eps,
        X=dW.size(0),
        Y=dW.size(1) if DIMS >= 2 else None,
        Z=dW.size(2) if DIMS == 3 else None,
    )
