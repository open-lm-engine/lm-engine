# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from .....kernels.custom_op import xma_op
from .state_update_kernel import _early_config_prune, _get_autotune_configs, _get_persistent_grid


@triton.autotune(
    configs=_get_autotune_configs(),
    key=[],
    prune_configs_by={"early_config_prune": _early_config_prune},
    reset_to_zero=["W_norm_ptr"],
)
@triton.jit
def _single_tensor_hyperball_weight_update_triton_kernel(
    u_ptr,
    u_stride,
    u_norm_ptr,
    lr,
    R_ptr,
    W_ptr,
    W_stride,
    W_norm_ptr,
    eps,
    X: tl.constexpr,
    Y: tl.constexpr,
    Z: tl.constexpr,
    COMPUTE_W_NORM: tl.constexpr,
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

    R = tl.load(R_ptr).to(tl.float32)

    u_norm = tl.load(u_norm_ptr).to(tl.float32)
    u_norm = tl.sqrt(u_norm)

    if COMPUTE_W_NORM:
        W_norm = 0.0
    else:
        W_norm = tl.load(W_norm_ptr).to(tl.float32)
        W_norm = tl.sqrt(W_norm)

    for TILE_ID in range(start, end):
        BLOCK_ID_X = TILE_ID % NUM_TILES_X
        TILE_ID_REM = TILE_ID // NUM_TILES_X

        BLOCK_X = BLOCK_ID_X * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
        MASK = BLOCK_X[:, None, None] < X

        W_ptrs = W_ptr + BLOCK_X[:, None, None] * W_stride[0]
        u_ptrs = u_ptr + BLOCK_X[:, None, None] * u_stride[0]

        if Y is not None:
            BLOCK_ID_Y = TILE_ID_REM % NUM_TILES_Y
            TILE_ID_REM = TILE_ID_REM // NUM_TILES_Y

            BLOCK_Y = BLOCK_ID_Y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
            MASK = MASK & (BLOCK_Y[None, :, None] < Y)

            W_ptrs += BLOCK_Y[None, :, None] * W_stride[1]
            u_ptrs += BLOCK_Y[None, :, None] * u_stride[1]

        if Z is not None:
            BLOCK_ID_Z = TILE_ID_REM % NUM_TILES_Z

            BLOCK_Z = BLOCK_ID_Z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
            MASK = MASK & (BLOCK_Z[None, None, :] < Z)

            W_ptrs += BLOCK_Z[None, None, :] * W_stride[2]
            u_ptrs += BLOCK_Z[None, None, :] * u_stride[2]

        u = tl.load(u_ptrs, mask=MASK).to(tl.float32)
        u /= u_norm + eps
        u *= lr * R

        W = tl.load(W_ptrs, mask=MASK).to(tl.float32)
        W -= u

        if COMPUTE_W_NORM:
            W_norm += tl.sum(W * W)
        else:
            W /= W_norm + eps
            W *= R
            tl.store(W_ptrs, W, mask=MASK)

    if COMPUTE_W_NORM:
        tl.atomic_add(W_norm_ptr, W_norm, sem="relaxed")


@xma_op(mutates_args={"W_norm"})
def _single_tensor_hyperball_weight_norm_triton(
    u: torch.Tensor,
    u_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    W: torch.Tensor,
    W_norm: torch.Tensor,
    eps: float,
    sm_margin: int,
) -> None:
    DIMS = W.dim()

    _single_tensor_hyperball_weight_update_triton_kernel[_get_persistent_grid(DIMS, sm_margin)](
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        lr=lr,
        R_ptr=R,
        W_ptr=W,
        W_stride=W.stride(),
        W_norm_ptr=W_norm,
        eps=eps,
        COMPUTE_W_NORM=True,
        X=W.size(0),
        Y=W.size(1) if DIMS >= 2 else None,
        Z=W.size(2) if DIMS == 3 else None,
    )


@xma_op(mutates_args={"W"})
def _single_tensor_hyperball_weight_update_triton(
    u: torch.Tensor,
    u_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    W: torch.Tensor,
    W_norm: torch.Tensor,
    eps: float,
    sm_margin: int,
) -> None:
    DIMS = W.dim()

    _single_tensor_hyperball_weight_update_triton_kernel[_get_persistent_grid(DIMS, sm_margin)](
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        lr=lr,
        R_ptr=R,
        W_ptr=W,
        W_stride=W.stride(),
        W_norm_ptr=W_norm,
        eps=eps,
        COMPUTE_W_NORM=False,
        X=W.size(0),
        Y=W.size(1) if DIMS >= 2 else None,
        Z=W.size(2) if DIMS == 3 else None,
    )
