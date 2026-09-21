# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .logging_utils import MetricsTrackingDict


MOE_ROUTER_AUX_LOSS = "moe_router_aux_loss"
MOE_Z_LOSS = "moe_z_loss"
MOE_EXPERT_FREQUENCY = "moe_expert_frequency"

_EXTRA_METRICS = MetricsTrackingDict({})


def reset_extra_metrics() -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = MetricsTrackingDict({})


def get_extra_metrics() -> MetricsTrackingDict:
    return _EXTRA_METRICS


def set_extra_metrics(metrics: MetricsTrackingDict) -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = metrics


def is_aux_loss_zero() -> bool:
    metrics_tracker = get_extra_metrics()

    for name in [MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS]:
        if name in metrics_tracker and metrics_tracker[name] is not None:
            return True

    return False
