# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .logging_utils import MetricsTrackingDict


MOE_ROUTER_AUX_LOSS = "moe_router_aux_loss"
MOE_Z_LOSS = "moe_z_loss"
MOE_EXPERT_FREQUENCY = "moe_expert_frequency"


class ExtraMetrics(MetricsTrackingDict):
    def is_aux_loss_zero(self) -> bool:
        for name in [MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS]:
            if name in self and self[name] is not None:
                return True

        return False

    def aggregate_loss(self) -> torch.Tensor:
        loss_aggregate = 0

        for key in self:
            is_loss = any([key.startswith(prefix) for prefix in [MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS]])
            assert is_loss or key.startswith(MOE_EXPERT_FREQUENCY)

            if is_loss:
                loss, coeff = self[key]
                loss_aggregate = loss_aggregate + loss * coeff

        return loss_aggregate


_EXTRA_METRICS = ExtraMetrics({})


def reset_extra_metrics() -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = MetricsTrackingDict({})


def get_extra_metrics() -> MetricsTrackingDict:
    return _EXTRA_METRICS


def set_extra_metrics(metrics: MetricsTrackingDict) -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = metrics
