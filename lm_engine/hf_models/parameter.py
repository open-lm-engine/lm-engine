# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn


_ALL_MARKERS = ["_no_weight_decay", "_has_mup_learning_rate", "_mup_learning_rate_divisor"]


def mark_parameter_as_no_weight_decay(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._no_weight_decay = True

    return parameter


def mark_parameter_as_mup_learning_rate(parameter: nn.Parameter | None, divisor: float | None = None) -> nn.Parameter | None:
    """Mark a parameter to be placed in an MuP learning-rate group.

    Args:
        parameter: the parameter to mark.
        divisor: optional float divisor to use for this parameter's learning rate.
            If None, the model-level `m_width` will be used (existing behaviour).

    Returns:
        The same parameter object (or None).
    """
    parameter._has_mup_learning_rate = True
    if divisor is not None:
        parameter._mup_learning_rate_divisor = float(divisor)
    return parameter


def get_mup_learning_rate_divisor(parameter: nn.Parameter | None) -> float | None:
    """Return a per-parameter MuP learning-rate divisor if set, else None."""
    if parameter is None:
        return None

    return getattr(parameter, "_mup_learning_rate_divisor", None)


def is_parameter_with_no_weight_decay(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_no_weight_decay", False)


def is_parameter_with_mup_learning_rate(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_has_mup_learning_rate", False)
