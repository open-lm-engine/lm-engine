# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .containers import ModelContainer
from .data import ResumableDataLoader, custom_iterator, get_next_batch
from .dtensors import dtensor_to_tensor
from .train_utils import all_reduce_metrics_tracker, track_metrics
from .utils import Accelerator, ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager


@torch.no_grad()
def evaluate(
    val_dataloader: ResumableDataLoader,
    model_container: ModelContainer,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
) -> MetricsTrackingDict:
    """main validation loop for the program

    Args:
        val_dataloader (ResumableDataLoader): validation dataloader
        model_container (ModelContainer): model container
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker

    Returns:
        MetricsTrackingDict: metrics tracker
    """

    if ProcessGroupManager.is_tensor_parallel_enabled():
        if ProcessGroupManager.is_tensor_parallel_first_rank():
            num_steps = 0 if val_dataloader is None else len(val_dataloader)
        else:
            num_steps = 0

        num_steps = torch.tensor(
            num_steps,
            device=Accelerator.get_current_device(),
            dtype=torch.int32 if Accelerator.get_accelerator() == Accelerator.trainium else torch.long,
        )

        torch.distributed.all_reduce(num_steps, group=ProcessGroupManager.get_tensor_parallel_group())
        num_steps = num_steps.item()
    else:
        num_steps = 0 if val_dataloader is None else len(val_dataloader)

    if num_steps == 0:
        return

    model_container.eval()

    metrics_tracker = MetricsTrackingDict({})
    val_dataloader = custom_iterator(val_dataloader, infinite=False)
    loss_tokens = 0

    for _ in range(num_steps):
        batch = get_next_batch(val_dataloader)
        loss_tokens += (batch["labels"] != -100).sum()
        loss_step_dict = model_container[0](batch)
        metrics_tracker = metrics_tracker + loss_step_dict

    metrics_tracker = metrics_tracker / loss_tokens.item()

    for key in metrics_tracker:
        metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

    metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    track_metrics(
        global_step=global_step,
        experiments_tracker=experiments_tracker,
        metrics_tracker=metrics_tracker,
        context="val",
    )

    model_container.train()

    return metrics_tracker
