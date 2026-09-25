# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .constants import MOE_EXPERT_FREQUENCY, MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS
from .logging_utils import MetricsTrackingDict


class ExtraMetrics(MetricsTrackingDict):
    def is_aux_loss_zero(self) -> bool:
        for key in self:
            is_loss = any(key.startswith(prefix) for prefix in [MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS])
            if is_loss and self[key] is not None:
                return False

        return True

    def aggregate_loss(self) -> torch.Tensor:
        loss_aggregate = 0

        for key in self:
            is_loss = self._is_loss(key)
            assert is_loss or key.startswith(MOE_EXPERT_FREQUENCY)

            if is_loss:
                loss, coeff = self[key]
                loss_aggregate = loss_aggregate + loss * coeff

        return loss_aggregate

    def get_metrics_for_logging(self) -> dict:
        # drops the coeff used for backprop so per-layer and total values logged here stay unweighted
        metrics = {}
        totals = {}

        for key in self:
            value = self[key]

            is_loss = self._is_loss(key)
            assert is_loss or key.startswith(MOE_EXPERT_FREQUENCY)

            if is_loss:
                loss, _ = value
                metrics[key] = loss

                prefix = MOE_ROUTER_AUX_LOSS if key.startswith(MOE_ROUTER_AUX_LOSS) else MOE_Z_LOSS
                totals[prefix] = totals.get(prefix, 0) + loss
            else:
                metrics[key] = value

        metrics.update(totals)

        return metrics

    def _is_loss(self, key: str) -> bool:
        for prefix in [MOE_ROUTER_AUX_LOSS, MOE_Z_LOSS]:
            if key.startswith(prefix):
                return True

        return False


_EXTRA_METRICS = ExtraMetrics({})


def reset_extra_metrics() -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = ExtraMetrics({})


def get_extra_metrics() -> MetricsTrackingDict:
    return _EXTRA_METRICS


def set_extra_metrics(metrics: MetricsTrackingDict) -> None:
    global _EXTRA_METRICS
    _EXTRA_METRICS = metrics
