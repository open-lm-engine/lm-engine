# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math


def _get_std_for_linear(
    initializer_range: float,
    init_method: str,
    m_width: float | None,
    fan_in: int | None,
    num_layers: int | None,
    use_depth_scaled_init: bool,
) -> float:
    if init_method == "mup":
        std = initializer_range / math.sqrt(m_width)
    elif init_method == "fan_in":
        std = 1 / math.sqrt(fan_in)
    elif init_method == "normal":
        std = initializer_range
    else:
        raise ValueError(f"unexpected init_method ({init_method})")

    if use_depth_scaled_init:
        std /= math.sqrt(2 * num_layers)

    return std


def _get_std_for_embedding(initializer_range: float, init_method: str, embed_dim: int) -> float:
    if init_method == "fan_in":
        std = 1 / math.sqrt(embed_dim)
    elif init_method in ["normal", "mup"]:
        std = initializer_range
    else:
        raise ValueError(f"unexpected init_method ({init_method})")

    return std
