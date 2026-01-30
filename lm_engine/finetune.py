# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import nullcontext

import torch
from torch.distributed.tensor.parallel import loss_parallel

from .arguments import TrainingArgs, get_args
from .checkpointing import ensure_last_checkpoint_is_saved, load_checkpoint_for_training, save_checkpoint
from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer, log_model_optimizer_container
from .data import ResumableDataLoader, custom_iterator, get_finetuning_dataloader, get_next_batch
from .distributed import wrap_model_container_for_distributed_training
from .dtensors import dtensor_to_tensor
from .enums import DatasetSplit, TuningMethod
from .hf_models import disable_generation_cache
from .kernels import enable_kernels
from .model_wrapper import get_model_container
from .optimization import get_learning_rate, get_optimizer_container, get_scheduler_container
from .pretrain import train_step_without_pipeline_parallel
from .train_utils import all_reduce_metrics_tracker, track_metrics
from .utils import (
    Accelerator,
    ExperimentsTracker,
    MetricsTrackingDict,
    ProcessGroupManager,
    StepTracker,
    TorchProfiler,
    init_distributed,
    set_seed,
    setup_tf32,
)


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

    forward_context = nullcontext
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

        num_steps = torch.tensor(num_steps, device=Accelerator.get_current_device(), dtype=torch.long)
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


def main() -> None:
    """main program"""

    setup_tf32()

    args: TrainingArgs = get_args(TrainingArgs)

    assert (
        args.tuning_args.tuning_method == TuningMethod.full_finetuning
    ), f"unexpected tuning method ({args.tuning_args.tuning_method})"

    # initialize distributed with nccl for multi-node communications
    init_distributed(
        tensor_parallel_world_size=args.distributed_args.tensor_parallel_world_size,
        pipeline_parallel_world_size=args.distributed_args.pipeline_parallel_world_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        zero_stage=args.distributed_args.stage,
        timeout_minutes=args.distributed_args.timeout_minutes,
        use_async_tensor_parallel=args.distributed_args.use_async_tensor_parallel,
    )

    StepTracker(
        micro_batch_size=args.training_parameters.micro_batch_size,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
    )

    set_seed(args.random_args.seed)

    assert args.distributed_args.num_pipeline_stages == 1, "pipeline parallel is not supported with finetuning"

    model_container = get_model_container(
        args, efficient_initialization=args.model_args.efficient_initialization, keep_in_fp32=True
    )

    train_dataloader = get_finetuning_dataloader(
        args, split=DatasetSplit.train, tokenizer=model_container[0].tokenizer
    )

    val_dataloader = None
    if args.training_parameters.eval_during_training:
        val_dataloader = get_finetuning_dataloader(
            args, split=DatasetSplit.val, tokenizer=model_container[0].tokenizer
        )

    model_container, _ = wrap_model_container_for_distributed_training(args, model_container)

    optimizer_container = get_optimizer_container(
        optimizer_class_name=args.optimizer_args.class_name,
        optimizer_class_args=args.optimizer_args.class_args,
        model_container=model_container,
        params_group_method=args.optimizer_args.params_group_method,
        use_optimizer_with_backward_hook=args.optimizer_args.use_optimizer_with_backward_hook,
    )

    lr_scheduler_container = get_scheduler_container(
        model_container=model_container,
        optimizer_container=optimizer_container,
        num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
        num_constant_steps=args.lr_scheduler_args.num_constant_steps,
        num_decay_steps=args.lr_scheduler_args.num_decay_steps,
        num_training_steps=args.training_parameters.num_training_steps,
        lr_decay_style=args.lr_scheduler_args.lr_decay_style,
        lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
        extra_lr_scheduler_args=args.lr_scheduler_args.extra_lr_scheduler_args,
        use_optimizer_with_backward_hook=args.optimizer_args.use_optimizer_with_backward_hook,
    )

    assert len(model_container) == len(optimizer_container)
    assert len(optimizer_container) == len(lr_scheduler_container)

    log_model_optimizer_container(model_container, optimizer_container)

    starting_iteration = 0
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, _, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, TrainingArgs, model_container, optimizer_container, lr_scheduler_container, train_dataloader
        )

    experiments_tracker = ExperimentsTracker(
        args.logging_args.experiments_tracker_name,
        args.logging_args.aim_args,
        args.logging_args.wandb_args,
        checkpoint_metadata=experiments_tracker_state_dict,
    )
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    with disable_generation_cache(), enable_kernels(args.kernel_args.kernels):
        train(
            args,
            model_container=model_container,
            optimizer_container=optimizer_container,
            lr_scheduler_container=lr_scheduler_container,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            experiments_tracker=experiments_tracker,
            starting_iteration=starting_iteration,
        )


if __name__ == "__main__":
    main()
