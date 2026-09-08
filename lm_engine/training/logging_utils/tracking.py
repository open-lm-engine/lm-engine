# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from ...accelerator import Accelerator
from ..containers import ModelContainer
from ..enums import ExperimentsTrackerName
from ..parallel import ProcessGroupManager
from ..utils import is_aim_available, is_wandb_available
from .loss_dict import MetricsTrackingDict


if TYPE_CHECKING:
    from ..arguments import BaseArgs


if is_aim_available():
    from aim import Run as AimRun

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


class ProgressBar:
    """progress bar for training or validation"""

    def __init__(self, start: int, end: int, desc: str | None = None) -> ProgressBar:
        self.is_tracking_rank = is_tracking_rank()
        if not self.is_tracking_rank:
            return

        self.progress_bar = tqdm(total=end, desc=desc)
        self.update(start)

    def update(self, n: int = 1) -> None:
        """updates progress bar

        Args:
            n (int, optional): Number of steps to update the progress bar with. Defaults to 1.
        """

        if not self.is_tracking_rank:
            return

        self.progress_bar.update(n=n)

    def track(self, **loss_kwargs) -> None:
        """track specific metrics in progress bar"""

        if not self.is_tracking_rank:
            return

        # for key in loss_kwargs:
        #     loss_kwargs[key] = "{0:.5f}".format(loss_kwargs[key])
        self.progress_bar.set_postfix(**loss_kwargs)


def get_code_provenance() -> dict:
    """Best-effort git commit/branch/modified for the lm-engine repo and the XMA submodule, so every
    run records exactly which kernel code it ran. Never raises — returns whatever it can resolve."""
    import os
    import subprocess

    def _git(path: str, *cmd: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", path, *cmd], stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except Exception:
            return None

    def _info(prefix: str, anchor_file: str, out: dict) -> None:
        # git -C on any path inside a repo resolves that repo's HEAD (the submodule has its own .git)
        path = os.path.dirname(os.path.abspath(anchor_file))
        commit = _git(path, "rev-parse", "HEAD")
        if commit is None:
            return
        out[f"{prefix}_commit"] = commit
        out[f"{prefix}_branch"] = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
        out[f"{prefix}_modified"] = bool(_git(path, "status", "--porcelain"))

    provenance: dict = {}
    _info("lm_engine", __file__, provenance)
    try:
        import xma

        _info("xma", xma.__file__, provenance)
    except Exception:
        pass
    return provenance


class ExperimentsTracker:
    """experiments tracker for training"""

    def __init__(
        self,
        experiments_tracker_name: ExperimentsTrackerName | None,
        aim_args: BaseArgs,
        wandb_args: BaseArgs,
        checkpoint_metadata: dict,
    ) -> ExperimentsTracker:
        self.is_tracking_rank = is_tracking_rank()
        self.experiments_tracker_name = experiments_tracker_name
        self.tracking_enabled = experiments_tracker_name is not None

        if not self.is_tracking_rank:
            return

        if experiments_tracker_name == ExperimentsTrackerName.aim:
            kwargs = aim_args.to_dict() if checkpoint_metadata is None else checkpoint_metadata
            self.run = AimRun(**kwargs)
        elif experiments_tracker_name == ExperimentsTrackerName.wandb:
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
        elif experiments_tracker_name is not None:
            raise ValueError(f"unexpected experiments_tracker ({experiments_tracker_name})")

    def log_args(self, args: BaseArgs, **extra_metadata) -> None:
        """log args

        Args:
            args (BaseArgs): pydantic object
        """

        if not self.is_tracking_rank:
            return

        if self.tracking_enabled:
            args: dict = args.to_dict()

            for k, v in get_code_provenance().items():
                args.setdefault(k, v)

            for k, v in extra_metadata.items():
                if k in args:
                    raise ValueError(f"duplicate key ({k})")
                args[k] = v

            if self.experiments_tracker_name == ExperimentsTrackerName.aim:
                for k, v in args.items():
                    try:
                        self.run[k] = v
                    except TypeError:
                        self.run[k] = str(v)
            elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
                wandb.config.update(args, allow_val_change=True)
            else:
                raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    def track(self, values: dict, step: int | None = None, context: str | None = None) -> None:
        """main tracking method

        Args:
            value (Any): value of the object to track
            step (int, optional): current step, auto-incremented if None. Defaults to None.
            context (str, optional): context for tracking. Defaults to None.
        """

        if not self.tracking_enabled:
            return

        if self.experiments_tracker_name == ExperimentsTrackerName.aim:
            if context is not None:
                context = {"subset": context}

            for key, value in values.items():
                self.run.track(value=value, name=key, step=step, context=context)
        elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
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
        else:
            raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    def finish(self) -> None:
        if not self.tracking_enabled or not self.is_tracking_rank:
            return

        if self.experiments_tracker_name == ExperimentsTrackerName.aim:
            self.run.close()
        elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
            wandb.finish()
        else:
            raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    def state_dict(self) -> dict:
        if not self.is_tracking_rank:
            return

        state_dict = {}
        if self.tracking_enabled:
            if self.experiments_tracker_name == ExperimentsTrackerName.aim:
                state_dict = {"run_hash": self.run.hash}
            elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
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
def track_parameter_and_gradient_info(
    model_container: ModelContainer,
    metrics_tracker: MetricsTrackingDict,
    gradient_clipping: float | None = None,
    gradient_norm: float | None = None,
    histograms: bool = False,
) -> None:
    assert is_wandb_available()
    assert len(model_container) == 1
    model = model_container[0]

    scale = 1.0
    if gradient_clipping is not None and gradient_norm is not None:
        total_norm = float(gradient_norm)
        scale = max(1.0, (total_norm + 1e-6) / gradient_clipping)

    def _maybe_gather_norm(tensor: torch.Tensor) -> float:
        norm = tensor.norm()
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        return norm.item()

    def _maybe_gather_tolist(tensor: torch.Tensor) -> list:
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        return tensor.detach().flatten().cpu().tolist()

    for name, param in model.named_parameters():
        metrics_tracker[f"param/norm/{name}"] = _maybe_gather_norm(param)
        if histograms:
            metrics_tracker[f"param/hist/{name}"] = wandb.Histogram(_maybe_gather_tolist(param))

        if param.grad is not None:
            metrics_tracker[f"grad/norm/{name}"] = _maybe_gather_norm(param.grad)
            metrics_tracker[f"scaled-grad/norm/{name}"] = _maybe_gather_norm(param.grad * scale)
            if histograms:
                metrics_tracker[f"grad/hist/{name}"] = wandb.Histogram(_maybe_gather_tolist(param.grad))
                metrics_tracker[f"scaled-grad/hist/{name}"] = wandb.Histogram(_maybe_gather_tolist(param.grad * scale))
