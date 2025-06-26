# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from tqdm import tqdm

from ..enums import ExperimentsTrackerName
from .packages import is_aim_available, is_wandb_available
from .parallel import is_tracking_rank
from .pydantic import BaseArgs


if is_aim_available():
    from aim import Run as AimRun

if is_wandb_available():
    import wandb


# to track the LSF/Slurm job in W&B per run - bobcalio
_JOB_ID = None if int(os.getenv("JOB_ID", -1)) == -1 else int(os.getenv("JOB_ID"))


class ProgressBar:
    """progress bar for training or validation"""

    def __init__(self, start: int, end: int, desc: str | None = None) -> None:
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


class ExperimentsTracker:
    """experiments tracker for training"""

    def __init__(
        self,
        experiments_tracker_name: ExperimentsTrackerName | None,
        aim_args: BaseArgs,
        wandb_args: BaseArgs,
        checkpoint_metadata: dict,
    ) -> None:
        self.is_tracking_rank = is_tracking_rank()
        if not self.is_tracking_rank:
            return

        self.experiments_tracker_name = experiments_tracker_name
        self.tracking_enabled = experiments_tracker_name is not None

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

    def log_args(self, args: BaseArgs) -> None:
        """log args

        Args:
            args (BaseArgs): pydantic object
        """

        if not self.is_tracking_rank:
            return

        if self.tracking_enabled:
            args: dict = args.to_dict()

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

        if not self.is_tracking_rank:
            return

        if self.tracking_enabled:
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
                wandb.log(values)
            else:
                raise ValueError(f"unexpected experiments_tracker ({self.experiments_tracker_name})")

    def finish(self) -> None:
        if not self.is_tracking_rank:
            return

        if self.tracking_enabled:
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
                    "entity": wandb.run.entity,
                    "project": wandb.run.project,
                }

        return state_dict
