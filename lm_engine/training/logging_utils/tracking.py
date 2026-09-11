# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

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


def _maybe_gather_norm(tensor: torch.Tensor) -> float:
    norm = tensor.norm()
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    return norm


@torch.no_grad()
def get_statistics_from_tensors(tensors: Iterable[tuple[str, torch.Tensor]], prefix: str) -> MetricsTrackingDict:
    total_norm = 0
    metrics_tracker = MetricsTrackingDict({})

    for name, tensor in tensors:
        norm = 0 if tensor is None else _maybe_gather_norm(tensor)
        metrics_tracker[f"{prefix}-norm/{name}"] = norm
        total_norm += norm**2

    metrics_tracker[f"{prefix}-norm/total"] = total_norm**0.5

    return metrics_tracker
