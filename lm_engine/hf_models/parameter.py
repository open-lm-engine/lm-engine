# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch.nn as nn


_INIT_MARKER = "_is_initialized"
_METADATA_MARKERS = ["_no_weight_decay", "_has_mup_learning_rate", "_per_row_hyperball"]
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


def mark_parameter_as_per_row_hyperball(parameter: nn.Parameter | None) -> nn.Parameter | None:
    # Routes this param through MuonHyperball's per-row path instead of Newton-Schulz: each row
    # is treated as an independent unit — the gradient is L2-normalized per row and the hyperball
    # projection is applied per row (each row gets its own radius). Used for conv kernels (one row
    # per output channel) and for 2D Linear weights where each row is a logical unit (e.g. DeltaMLP
    # b_proj, where row h is head h's β-projection).
    if parameter is not None:
        parameter._per_row_hyperball = True

    return parameter


def set_split_spec(
    parameter: nn.Parameter | None,
    tag: str,
    shape: tuple[int, ...],
    **axes: int,
) -> nn.Parameter | None:
    if parameter is not None:
        parameter._split_spec = (tag, tuple(shape), dict(axes))

    return parameter


def get_split_spec(parameter: nn.Parameter | None) -> tuple | None:
    return getattr(parameter, "_split_spec", None)


def is_parameter_with_no_weight_decay(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_no_weight_decay", False)


def is_parameter_with_mup_learning_rate(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_has_mup_learning_rate", False)


def is_parameter_per_row_hyperball(parameter: nn.Parameter | None) -> bool:
    return getattr(parameter, "_per_row_hyperball", False)


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
            spec = getattr(param, "_split_spec", None)
            if spec is not None:
                marker_maps[-1][param_name]["_split_spec"] = spec

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
