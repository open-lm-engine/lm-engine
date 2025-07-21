# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn


_ALL_MARKERS = ["_no_weight_decay", "_has_mup_learning_rate"]


def mark_parameter_as_no_weight_decay(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._no_weight_decay = True

    return parameter


def mark_parameter_as_mup_learning_rate(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._has_mup_learning_rate = True

    return parameter


def is_parameter_with_no_weight_decay(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_no_weight_decay", False)


def is_parameter_with_mup_learning_rate(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_has_mup_learning_rate", False)
