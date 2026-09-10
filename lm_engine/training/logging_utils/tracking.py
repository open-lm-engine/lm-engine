# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from ...accelerator import Accelerator
from ..parallel import ProcessGroupManager
from ..utils import is_wandb_available


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


# to track the LSF/Slurm job in W&B per run - bobcalio
_JOB_ID = None if int(os.getenv("JOB_ID", -1)) == -1 else int(os.getenv("JOB_ID"))


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
        # track the LSF/Slurm job in W&B per run - bobcalio
        if _JOB_ID is not None:
            wandb.define_metric("job", step_metric="iteration", hidden=True, step_sync=True)

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
        # track the LSF/Slurm job in W&B per run - bobcalio
        if _JOB_ID is not None:
            values["job"] = _JOB_ID

        # FIXME this is needed to prevent TPU from getting stuck
        # on GPU, only 1 rank needs to call this but on TPUs, every rank needs to call this
        if Accelerator.get_accelerator() == Accelerator.tpu:
            values = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in values.items()}

        if self.is_tracking_rank:
            wandb.log(values)

    def finish(self) -> None:
        if not self.tracking_enabled or not self.is_tracking_rank:
            return

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
