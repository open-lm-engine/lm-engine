# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from contextlib import nullcontext

import torch
from torch.distributed.tensor.parallel import loss_parallel

from .arguments import TrainingArgs
from .checkpointing import ensure_last_checkpoint_is_saved, save_checkpoint
from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer
from .data import ResumableDataLoader, custom_iterator, get_next_batch
from .dtensors import dtensor_to_tensor
from .optimization import get_learning_rate
from .pretrain import train_step_without_pipeline_parallel
from .train_utils import all_reduce_metrics_tracker, track_metrics
from .utils import Accelerator, ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager, TorchProfiler


def train(
    args: TrainingArgs,
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
    val_dataloader: ResumableDataLoader,
    experiments_tracker: ExperimentsTracker,
    starting_iteration: int = 0,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model_container (ModelContainer): container of models
        pipeline_schedule (_PipelineSchedule): pipeline schedule
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
        train_dataloader (ResumableDataLoader): training dataloader
        val_dataloader (ResumableDataLoader): validation dataloader
        experiments_tracker (ExperimentsTracker): metrics tracker
        starting_iteration (int): starting iteration
    """

    num_training_steps = args.training_parameters.num_training_steps
    gradient_clipping = args.training_parameters.gradient_clipping

    eval_during_training = args.training_parameters.eval_during_training
    eval_interval = args.training_parameters.eval_interval
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    model_container.train()

    # need this for iterating infinitely
    train_dataloader_infinite = custom_iterator(train_dataloader, infinite=True)

    if eval_during_training:
        evaluate(val_dataloader, model_container, starting_iteration, experiments_tracker)

    forward_context = loss_parallel if ProcessGroupManager.is_tensor_parallel_enabled() else nullcontext
    backward_context = loss_parallel if ProcessGroupManager.is_tensor_parallel_enabled() else nullcontext

    torch_profiler = TorchProfiler(args.logging_args.torch_profiler_trace_path)
    torch_profiler.__enter__()

    metrics_tracker = MetricsTrackingDict({})

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1

        loss_step_dict = train_step_without_pipeline_parallel(
            model_container=model_container,
            optimizer_container=optimizer_container,
            lr_scheduler_container=lr_scheduler_container,
            train_dataloader=train_dataloader_infinite,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
            sync_every_gradient_accumulation_step=args.distributed_args.sync_every_gradient_accumulation_step,
            lm_loss_multiplier=None,
            tuning_method=args.tuning_args.tuning_method,
        )

        metrics_tracker = metrics_tracker + loss_step_dict
        torch_profiler.step()

        if global_step % log_interval == 0:
            metrics_tracker = metrics_tracker / log_interval
            metrics_tracker["learning_rate"] = get_learning_rate(model_container, lr_scheduler_container)

            track_metrics(
                global_step=global_step,
                experiments_tracker=experiments_tracker,
                metrics_tracker=metrics_tracker,
                context="train",
            )

            metrics_tracker = MetricsTrackingDict({})

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(val_dataloader, model_container, global_step, experiments_tracker)

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(
                args=args,
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader,
                experiments_tracker=experiments_tracker,
                iteration=global_step,
            )

    ensure_last_checkpoint_is_saved()
    torch_profiler.__exit__(None, None, None)


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


if __name__ == "__main__":
    main()
