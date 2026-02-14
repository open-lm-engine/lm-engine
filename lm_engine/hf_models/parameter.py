# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn


_INIT_MARKER = "_is_initialized"
_METADATA_MARKERS = ["_no_weight_decay", "_has_mup_learning_rate"]
_ALL_MARKERS = _METADATA_MARKERS + [_INIT_MARKER]


def mark_parameter_as_no_weight_decay(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._no_weight_decay = True

    return parameter


def mark_parameter_as_mup_learning_rate(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._has_mup_learning_rate = True

    return parameter


def mark_parameter_as_initialized(parameter: nn.Parameter | None) -> nn.Parameter | None:
    if parameter is not None:
        parameter._is_initialized = True

    return parameter


def is_parameter_with_no_weight_decay(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_no_weight_decay", False)


def is_parameter_with_mup_learning_rate(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_has_mup_learning_rate", False)


def is_parameter_initialized(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, _INIT_MARKER, False)


def get_parameter_marker_maps(model_container: list[nn.Module], extra_markers: list[str] = []) -> list[dict]:
    if isinstance(model_container, nn.Module):
        model_container = [model_container]

    marker_maps = []
    for model in model_container:
        marker_maps.append({})
        for param_name, param in model.named_parameters():
            marker_maps[-1][param_name] = {}
            for marker in _METADATA_MARKERS + extra_markers:
                marker_maps[-1][param_name][marker] = getattr(param, marker, False)

    return marker_maps


def set_parameter_marker_maps(
    model_container: list[nn.Module],
    marker_maps: list[dict],
    replacement_patterns: list[tuple[str]] = [],
    _trim_prefix: str | None = None,
) -> None:
    if isinstance(model_container, nn.Module):
        model_container = [model_container]

    for model, _marker_map in zip(model_container, marker_maps):
        for param_name, parameter in model.named_parameters():
            for pattern, replacement in replacement_patterns:
                param_name = param_name.replace(pattern, replacement)

            if _trim_prefix is not None:
                param_name = param_name.removeprefix(_trim_prefix)

            for marker, value in _marker_map[param_name].items():
                setattr(parameter, marker, value)
