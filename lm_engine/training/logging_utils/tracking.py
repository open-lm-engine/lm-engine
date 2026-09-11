# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from ...accelerator import Accelerator
from ..containers import ModelContainer
from ..parallel import ProcessGroupManager
from ..utils import is_wandb_available
from .loss_dict import MetricsTrackingDict


if TYPE_CHECKING:
    from ..arguments import BaseArgs


if is_wandb_available():
    import wandb


def is_tracking_rank() -> bool:
    return (
        ProcessGroupManager.get_data_parallel_rank() == 0
        and ProcessGroupManager.get_context_parallel_rank() == 0
        and ProcessGroupManager.is_tensor_parallel_first_rank()
        and ProcessGroupManager.get_pipeline_parallel_rank()
        == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
    )


# track the Slurm job in W&B per run
_JOB_ID = os.getenv("SLURM_JOB_ID")


class ExperimentsTracker:
    """experiments tracker for training"""

    def __init__(
        self,
        wandb_args: BaseArgs | None,
        checkpoint_metadata: dict,
    ) -> ExperimentsTracker:
        self.is_tracking_rank = is_tracking_rank()
        self.tracking_enabled = wandb_args is not None

        if not self.is_tracking_rank or not self.tracking_enabled:
            return

        kwargs = wandb_args.to_dict() if checkpoint_metadata is None else checkpoint_metadata
        resume = None if checkpoint_metadata is None else "auto"

        wandb.init(resume=resume, **kwargs)

        # this is for a custom step, we can't use the wandb step
        # since it doesn't allow time travel to the past
        wandb.define_metric("iteration", hidden=True)
        if _JOB_ID is not None:
            wandb.define_metric("job_id", step_metric="iteration", step_sync=True)

        wandb.define_metric("train/*", step_metric="iteration", step_sync=True)
        wandb.define_metric("val/*", step_metric="iteration", step_sync=True)

    def log_args(self, args: BaseArgs, **extra_metadata) -> None:
        """log args

        Args:
            args (BaseArgs): pydantic object
        """

        if not self.is_tracking_rank:
            return

        if self.tracking_enabled:
            args: dict = args.to_dict()

            for k, v in extra_metadata.items():
                if k in args:
                    raise ValueError(f"duplicate key ({k})")
                args[k] = v

            wandb.config.update(args, allow_val_change=True)

    def track(self, values: dict, step: int | None = None, context: str | None = None) -> None:
        """main tracking method

        Args:
            value (Any): value of the object to track
            step (int, optional): current step, auto-incremented if None. Defaults to None.
            context (str, optional): context for tracking. Defaults to None.
        """

        if not self.tracking_enabled:
            return

        if context is not None:
            values = {f"{context}/{k}": v for k, v in values.items()}

        # this is for a custom step, we can't use the wandb step
        # since it doesn't allow time travel to the past
        values["iteration"] = step
        if _JOB_ID is not None:
            values["job_id"] = _JOB_ID

        # FIXME this is needed to prevent TPU from getting stuck
        # on GPU, only 1 rank needs to call this but on TPUs, every rank needs to call this
        if Accelerator.get_accelerator() == Accelerator.tpu:
            values = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in values.items()}

        if self.is_tracking_rank:
            wandb.log(values)

    def finish(self) -> None:
        if self.tracking_enabled and self.is_tracking_rank:
            wandb.finish()

    def state_dict(self) -> dict:
        if not self.is_tracking_rank:
            return

        state_dict = {}
        if self.tracking_enabled:
            state_dict = {
                "id": wandb.run.id,
                "name": wandb.run.name,
                "tags": wandb.run.tags,
                "group": wandb.run.group,
                "notes": wandb.run.notes,
                "entity": wandb.run.entity,
                "project": wandb.run.project,
            }

        return state_dict


@torch.no_grad()
def compute_model_statistics(model_container: ModelContainer) -> MetricsTrackingDict:
    assert len(model_container) == 1
    model = model_container[0]

    metrics_tracker = MetricsTrackingDict({})

    def _maybe_gather_norm(tensor: torch.Tensor) -> float:
        norm = tensor.norm()
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        return norm

    param_norm_squared_sum = 0
    grad_norm_squared_sum = 0

    for name, param in model.named_parameters():
        param_norm = _maybe_gather_norm(param)
        metrics_tracker[f"param/norm/{name}"] = param_norm
        param_norm_squared_sum += param_norm**2

        if param.grad is None:
            continue

        grad_norm = _maybe_gather_norm(param.grad)
        metrics_tracker[f"grad/norm/{name}"] = grad_norm

        grad_norm_squared_sum += grad_norm**2

    metrics_tracker["param/total-norm"] = param_norm_squared_sum**0.5
    metrics_tracker["grad/total-norm"] = grad_norm_squared_sum**0.5

    return metrics_tracker
