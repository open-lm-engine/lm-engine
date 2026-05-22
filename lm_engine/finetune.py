# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .containers import ModelContainer
from .data import ResumableDataLoader, custom_iterator, get_next_batch
from .utils import Accelerator, MetricsTrackingDict, ProcessGroupManager


@torch.no_grad()
def evaluate(val_dataloader: ResumableDataLoader, model_container: ModelContainer) -> MetricsTrackingDict:
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
