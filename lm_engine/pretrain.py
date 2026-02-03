# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
from git import Repo
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.data import DataLoader

from .arguments import DistillationArgs, TrainingArgs, get_args
from .checkpointing import ensure_last_checkpoint_is_saved, load_checkpoint_for_training, save_checkpoint
from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer, log_model_optimizer_container
from .data import ResumableDataLoader, get_next_batch, get_pretraining_dataloaders
from .distributed import wrap_model_container_for_distributed_training
from .dtensors import dtensor_to_tensor
from .enums import TuningMethod
from .hf_models import disable_generation_cache
from .kernels import enable_kernels
from .model_wrapper import broadcast_tensor_parallel_input, get_model_container
from .optimization import get_learning_rate, get_optimizer_container, get_scheduler_container
from .train_utils import all_reduce_metrics_tracker, get_model_tflops, track_metrics
from .utils import (
    Accelerator,
    Communication,
    ExperimentsTracker,
    MetricsTrackingDict,
    ProcessGroupManager,
    StepTracker,
    TorchProfiler,
    init_distributed,
    is_torch_xla_available,
    is_torchao_available,
    log_environment,
    log_rank_0,
    set_seed,
    setup_tf32,
)


if is_torch_xla_available():
    from torch_xla import launch as xla_launch
    from torch_xla import step as xla_step
    from torch_xla import sync as xla_sync

if is_torchao_available():
    from .distributed import FP8Manager


def train_step_with_pipeline_parallel(
    model_container: ModelContainer,
    pipeline_schedule: _PipelineSchedule,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
    gradient_clipping: float,
    sequence_length: int,
) -> MetricsTrackingDict:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model_container (ModelContainer): container of models
        pipeline_schedule (_PipelineSchedule): pipeline schedule
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_clipping (float): gradient clipping value
        sequence_length (int): sequence length

    Returns:
        MetricsTrackingDict: metrics to track
    """

    fsdp_algorithm = 2 if hasattr(model_container[0], "set_requires_gradient_sync") else 1
    grad_norm = []

    optimizer_container.zero_grad()

    batch = get_next_batch(train_dataloader)

    if ProcessGroupManager.is_tensor_parallel_first_rank():
        batch = batch["text"]
        batch = batch.to(Accelerator.get_current_device())

    if ProcessGroupManager.is_tensor_parallel_enabled():
        batch = broadcast_tensor_parallel_input(batch, (StepTracker.get_local_batch_size(), sequence_length + 1))

    is_first_pipeline_rank = ProcessGroupManager.get_pipeline_parallel_rank() == 0
    is_last_pipeline_rank = (
        ProcessGroupManager.get_pipeline_parallel_rank() == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
    )

    if is_first_pipeline_rank:
        pipeline_schedule.step(batch)
    elif is_last_pipeline_rank:
        losses = []
        labels = batch[:, 1:]
        pipeline_schedule.step(target=labels, losses=losses)
    else:
        pipeline_schedule.step()

    if gradient_clipping is not None:
        for model in model_container:
            if fsdp_algorithm == 1:
                grad_norm.append(model.clip_grad_norm_(gradient_clipping))
            else:
                grad_norm.append(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping))

    if is_torchao_available():
        FP8Manager.sync_float8_amax_and_scale_history(model_container)

    optimizer_container.step()
    lr_scheduler_container.step()

    if is_torchao_available():
        FP8Manager.precompute_float8_dynamic_scale_for_fsdp(model_container)

    metrics_tracker = MetricsTrackingDict({})

    with torch.inference_mode():
        if gradient_clipping is not None:
            grad_norm = dtensor_to_tensor(sum(grad_norm))

        torch.distributed.all_reduce(grad_norm, group=ProcessGroupManager.get_pipeline_parallel_group())

        if is_last_pipeline_rank:
            losses = sum(losses)
            losses = losses.squeeze(0)

            metrics_tracker = metrics_tracker + {"loss": losses, "grad_norm": grad_norm}
            metrics_tracker = metrics_tracker + model.get_extra_metrics()
            model.reset_extra_metrics()

            metrics_tracker = metrics_tracker / StepTracker.get_gradient_accumulation_steps()

            if gradient_clipping is not None:
                metrics_tracker["grad_norm"] = grad_norm

            for key in metrics_tracker:
                metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

            metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    return metrics_tracker


def train_step_without_pipeline_parallel(
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
    gradient_clipping: float,
    forward_context: AbstractContextManager,
    backward_context: AbstractContextManager,
    sync_every_gradient_accumulation_step: bool,
    lm_loss_multiplier: float,
    tuning_method: TuningMethod,
) -> MetricsTrackingDict:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model_container (ModelContainer): container of models
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_clipping (float): gradient clipping value
        forward_context (AbstractContextManager): a context that is used for every model forward call
        backward_context (AbstractContextManager): a context that is used for every model backward call
        sync_every_gradient_accumulation_step (bool): whether to sync on every gradient accumulation step
        lm_loss_multiplier (int): lm loss multiplier
        tuning_method (TuningMethod): tuning method for the current run

    Returns:
        MetricsTrackingDict: metrics to track
    """

    assert len(model_container) == 1
    model = model_container[0]

    fsdp_algorithm = 2 if hasattr(model, "set_requires_gradient_sync") else 1

    no_sync = nullcontext
    if not sync_every_gradient_accumulation_step:
        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        else:
            model.set_requires_gradient_sync(False)

    metrics_tracker = MetricsTrackingDict({})
    optimizer_container.zero_grad()

    gradient_accumulation_steps = StepTracker.get_gradient_accumulation_steps()

    if tuning_method == TuningMethod.full_finetuning:
        assert lm_loss_multiplier is None

        # note the effect of gradient accumulation division is already in the lm_loss_multiplier
        batches = [get_next_batch(train_dataloader) for _ in range(gradient_accumulation_steps)]
        lm_loss_multiplier = gradient_accumulation_steps / sum([(batch["labels"] != -100).sum() for batch in batches])
    else:
        batches = None

    accelerator = Accelerator.get_accelerator()

    with (xla_step if accelerator == Accelerator.tpu else nullcontext)():
        with no_sync():
            for step in range(gradient_accumulation_steps - 1):
                with forward_context():
                    batch = get_next_batch(train_dataloader) if batches is None else batches[step]
                    loss_micro_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)

                with backward_context():
                    loss_micro_step: torch.Tensor = loss_micro_step_dict["loss"] / gradient_accumulation_steps
                    loss_micro_step.backward()

                with torch.inference_mode():
                    metrics_tracker = metrics_tracker + loss_micro_step_dict

        if fsdp_algorithm == 2:
            model.set_requires_gradient_sync(True)

        with forward_context():
            batch = get_next_batch(train_dataloader) if batches is None else batches[-1]
            loss_micro_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)

        with backward_context():
            loss_micro_step: torch.Tensor = loss_micro_step_dict["loss"] / gradient_accumulation_steps
            loss_micro_step.backward()

        with torch.inference_mode():
            metrics_tracker = metrics_tracker + loss_micro_step_dict

        if gradient_clipping is None:
            grad_norm = None
        elif fsdp_algorithm == 1:
            grad_norm = model.clip_grad_norm_(gradient_clipping)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        if is_torchao_available():
            FP8Manager.sync_float8_amax_and_scale_history([model])

        optimizer_container.step()
        lr_scheduler_container.step()

    if accelerator == Accelerator.tpu:
        xla_sync()

    if is_torchao_available():
        FP8Manager.precompute_float8_dynamic_scale_for_fsdp([model])

    with torch.inference_mode():
        metrics_tracker = metrics_tracker / gradient_accumulation_steps
        metrics_tracker["grad_norm"] = (
            torch.zeros((), device=Accelerator.get_current_device(), dtype=torch.float32)
            if grad_norm is None
            else grad_norm
        )

        for key in metrics_tracker:
            metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    return metrics_tracker


def track_val_metrics(
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    metrics_tracker: MetricsTrackingDict,
    group_name: str | None = None,
    context: str = "val",
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): experiments tracker
        metrics_tracker (MetricsTrackingDict): metrics tracker
        group_name (str | None): group name for the validation / test set
        context (str): context
    """

    message = f"step = {global_step}"
    if group_name is not None:
        message += f", group_name = {group_name}"

    for key in metrics_tracker:
        message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_rank_0(logging.INFO, message)

    if group_name is None:
        message = metrics_tracker.get_dict()
    else:
        message = {}
        for key in metrics_tracker:
            message[f"{group_name}-{key}"] = metrics_tracker[key]

    experiments_tracker.track(message, step=global_step, context=context)


def train(
    args: TrainingArgs,
    model_container: ModelContainer,
    pipeline_schedule: _PipelineSchedule,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: DataLoader,
    val_dataloaders: list[DataLoader],
    test_dataloaders: list[DataLoader],
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
        train_dataloader (DataLoader): training dataloader
        val_dataloaders (list[DataLoader]): validation dataloaders
        test_dataloaders (list[DataLoader]): test dataloaders
        experiments_tracker (ExperimentsTracker): metrics tracker
        starting_iteration (int): starting iteration
    """

    num_training_steps = args.training_parameters.num_training_steps
    gradient_clipping = args.training_parameters.gradient_clipping

    eval_during_training = args.training_parameters.eval_during_training
    eval_interval = args.training_parameters.eval_interval
    save_interval = args.save_args.save_interval
    log_interval = args.logging_args.log_interval

    val_weighted_split_paths = args.datasets[0].class_args.get("val_weighted_split_paths")
    group_names = [None]
    if val_weighted_split_paths is not None:
        group_names = [key for key in val_weighted_split_paths.keys()[0]]

    model_container.train()

    micro_batch_size = args.training_parameters.micro_batch_size
    sequence_length = args.datasets[0].class_args.get("sequence_length")
    global_batch_size = StepTracker.get_global_batch_size()
    tokens_per_batch = global_batch_size * sequence_length

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(
            val_dataloaders,
            model_container,
            starting_iteration,
            experiments_tracker,
            eval_steps,
            group_names,
            lm_loss_multiplier=1 / (micro_batch_size * sequence_length),
            context="val",
        )

    is_pipeline_parallel_enabled = args.distributed_args.num_pipeline_stages > 1
    if not is_pipeline_parallel_enabled:
        assert len(model_container) == 1

    # model flops per accelerator
    model_flops = (
        get_model_tflops(
            config=model_container[0].config,
            batch_size=global_batch_size,
            sequence_length=sequence_length,
            gradient_checkpointing_method=args.distributed_args.gradient_checkpointing_method,
            gradient_checkpointing_args=args.distributed_args.gradient_checkpointing_args,
        )
        / ProcessGroupManager.get_world_size()
    )

    forward_context = nullcontext
    backward_context = loss_parallel if ProcessGroupManager.is_tensor_parallel_enabled() else nullcontext

    torch_profiler = TorchProfiler(args.logging_args.torch_profiler_trace_path)
    torch_profiler.__enter__()

    start_time = time.perf_counter()
    steps_since_start_time = 0
    metrics_tracker = MetricsTrackingDict({})

    global_step = starting_iteration
    while global_step < num_training_steps:
        global_step += 1
        steps_since_start_time += 1

        if is_pipeline_parallel_enabled:
            loss_step_dict = train_step_with_pipeline_parallel(
                model_container=model_container,
                pipeline_schedule=pipeline_schedule,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader,
                gradient_clipping=gradient_clipping,
                sequence_length=sequence_length,
            )
        else:
            loss_step_dict = train_step_without_pipeline_parallel(
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader,
                gradient_clipping=gradient_clipping,
                forward_context=forward_context,
                backward_context=backward_context,
                sync_every_gradient_accumulation_step=args.distributed_args.sync_every_gradient_accumulation_step,
                lm_loss_multiplier=1 / (micro_batch_size * sequence_length),
                tuning_method=args.tuning_args.tuning_method,
            )

        metrics_tracker = metrics_tracker + loss_step_dict
        torch_profiler.step()

        if global_step % log_interval == 0:
            metrics_tracker = metrics_tracker / log_interval

            time_elapsed = time.perf_counter() - start_time
            step_time = time_elapsed / steps_since_start_time

            metrics_tracker["learning_rate"] = get_learning_rate(model_container, lr_scheduler_container)

            if model_flops is not None:
                metrics_tracker["FLOPs"] = model_flops * steps_since_start_time / time_elapsed

            metrics_tracker["billion_tokens_per_day"] = tokens_per_batch * 86400 / step_time / 1e9
            metrics_tracker["step_time (sec)"] = step_time

            track_metrics(
                global_step=global_step,
                experiments_tracker=experiments_tracker,
                metrics_tracker=metrics_tracker,
                context="train",
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0
            metrics_tracker = MetricsTrackingDict({})

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(
                val_dataloaders,
                model_container,
                global_step,
                experiments_tracker,
                eval_steps,
                group_names,
                lm_loss_multiplier=1 / (micro_batch_size * sequence_length),
                context="val",
            )

        if global_step % save_interval == 0 or global_step == num_training_steps:
            save_checkpoint(
                args=args,
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=None,
                experiments_tracker=experiments_tracker,
                iteration=global_step,
                metadata={
                    "consumed_samples": global_step * global_batch_size,
                    "commit_id": Repo(Path(__file__).parents[1]).git.rev_parse("HEAD"),
                },
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        evaluate(
            test_dataloaders,
            model_container,
            global_step,
            experiments_tracker,
            eval_steps,
            group_names,
            lm_loss_multiplier=1 / (micro_batch_size * sequence_length),
            context="test",
        )

    ensure_last_checkpoint_is_saved()
    torch_profiler.__exit__(None, None, None)


@torch.no_grad()
def evaluate(
    val_dataloaders: list[DataLoader],
    model_container: ModelContainer,
    global_step: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int,
    group_names: list[str],
    lm_loss_multiplier: float,
    context: str,
) -> float:
    """main validation loop for the program

    Args:
        val_dataloaders (list[DataLoader]): list of validation dataloaders
        model_container (ModelContainer): container of models
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (list[str]): names of the datasets in validation/test group
        lm_loss_multiplier (float): lm loss multiplier
        context (str): context

    Returns:
        MetricsTrackingDict: metrics tracker
    """

    assert len(model_container) == 1
    model = model_container[0]

    if ProcessGroupManager.is_tensor_parallel_enabled():
        # other tensor parallel ranks need to be told if val dataloader is None or not
        is_val_dataloader_none = (
            val_dataloaders is None or len(val_dataloaders) == 0
            if ProcessGroupManager.is_tensor_parallel_first_rank()
            else None
        )
        is_val_dataloader_none = Communication.broadcast_object(
            is_val_dataloader_none,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )
    else:
        is_val_dataloader_none = val_dataloaders is None or len(val_dataloaders) == 0

    if is_val_dataloader_none:
        return

    model.eval()

    for group_name, val_dataloader in zip(group_names, val_dataloaders):
        metrics_tracker = MetricsTrackingDict({})

        for _ in range(eval_steps):
            batch = get_next_batch(val_dataloader)
            loss_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)
            metrics_tracker = metrics_tracker + loss_step_dict

        metrics_tracker = metrics_tracker / eval_steps

        for key in metrics_tracker:
            metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

        track_val_metrics(
            global_step=global_step,
            experiments_tracker=experiments_tracker,
            metrics_tracker=metrics_tracker,
            group_name=group_name,
            context=context,
        )

    model.train()

    return metrics_tracker


def main(args_class: type[DistillationArgs | TrainingArgs] = TrainingArgs) -> None:
    """main program"""

    setup_tf32()

    args: TrainingArgs | DistillationArgs = get_args(args_class)

    if args_class == TrainingArgs:
        assert (
            args.tuning_args.tuning_method == TuningMethod.pretraining
        ), f"unexpected tuning method ({args.tuning_args.tuning_method})"
    elif args_class == DistillationArgs:
        assert args.distributed_args.fsdp_algorithm == 2, "Distillation is only supported with FSDP-2"

        assert (
            args.tuning_args.tuning_method == TuningMethod.distillation
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

    args.log_args()
    log_environment()

    StepTracker(
        micro_batch_size=args.training_parameters.micro_batch_size,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
    )

    set_seed(args.random_args.seed)

    if args_class == DistillationArgs:
        assert args.distributed_args.num_pipeline_stages == 1, "pipeline parallel is not supported with distillation"

    model_container = get_model_container(
        args, efficient_initialization=args.model_args.efficient_initialization, keep_in_fp32=True
    )

    model_container, pipeline_schedule = wrap_model_container_for_distributed_training(args, model_container)

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
    metadata = {}
    experiments_tracker_state_dict = None
    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, args_class, model_container, optimizer_container, lr_scheduler_container, None
        )

        # metadata field contains the dataloader state so we need to reset it here
        if not args.load_args.load_dataloader_state:
            metadata["consumed_samples"] = 0

    train_dataloader, val_dataloaders, test_dataloaders = get_pretraining_dataloaders(
        args, model_container[0].tokenizer, metadata.get("consumed_samples", 0)
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
            pipeline_schedule=pipeline_schedule,
            optimizer_container=optimizer_container,
            lr_scheduler_container=lr_scheduler_container,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            test_dataloaders=test_dataloaders,
            experiments_tracker=experiments_tracker,
            starting_iteration=starting_iteration,
        )


def xla_main(*args):
    main()


if __name__ == "__main__":
    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.tpu:
        xla_launch(xla_main)
    else:
        main()
