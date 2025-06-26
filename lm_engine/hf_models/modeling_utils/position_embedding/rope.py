# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""Logic is copied from transformers.models.llama.modeling_utils with slight modifications"""

import math

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mscale = 1

        self.reset_parameters()

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=dtype)

        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)

        return cos, sin

    def reset_parameters(self) -> None:
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, dtype=torch.float32)

    @torch.no_grad()
    def _set_cos_sin_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len

        inv_freq = self._get_inv_freq()
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        device = self.cos_cached.device if hasattr(self, "cos_cached") else None

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(device=device, dtype=dtype), persistent=False)

    def _get_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))


class YaRNScaledRoPE(RoPE):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scale: float = 1,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        nn.Module.__init__(self)

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = _yarn_get_mscale(self.scale) * self.attn_factor

        self.reset_parameters()

    def _get_inv_freq(self) -> torch.Tensor:
        pos_freqs = self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.head_dim, self.base, self.original_max_position_embeddings
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.head_dim // 2).float()
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        return inv_freq


def apply_rotary_pos_emb(
    x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos_sin

    head_dim = x.size(-1)
    rope_dim = cos.size(-1)

    if head_dim == rope_dim:
        x = (x * cos) + (_rotate_half(x) * sin)
    elif rope_dim < head_dim:
        x_nope, x_rope = x.split((head_dim - rope_dim, rope_dim), dim=-1)
        x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
        x = torch.cat([x_nope, x_rope], dim=-1)
    else:
        raise ValueError("rope_dim should be less than head_dim")

    return x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: int = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: int = 10000, max_position_embeddings: int = 2048
) -> int:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min: float, max: float, dim: int) -> torch.Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
