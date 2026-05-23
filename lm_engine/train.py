# **************************************************
# Copyright (c) 2026, Mayank Mishra
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

from .accelerator import Accelerator
from .arguments import DistillationArgs, TrainingArgs, get_args
from .checkpointing import ensure_last_checkpoint_is_saved, load_checkpoint_for_training, save_checkpoint
from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer, log_model_optimizer_container
from .data import (
    DatasetSplit,
    ResumableDataLoader,
    custom_iterator,
    get_finetuning_dataloader,
    get_next_batch,
    get_pretraining_dataloaders,
)
from .dtensors import dtensor_to_tensor
from .enums import TuningMethod
from .hf_models import disable_generation_cache
from .kernels import enable_kernels
from .logging_utils import (
    ExperimentsTracker,
    MetricsTrackingDict,
    StepTracker,
    TorchProfiler,
    log_environment,
    log_rank_0,
)
from .model_wrapper import broadcast_tensor_parallel_input, get_model_container
from .optimization import get_learning_rate, get_optimizer_container, get_scheduler_container
from .parallel import ProcessGroupManager, wrap_model_container_for_distributed_training
from .train_utils import all_reduce_metrics_tracker, get_model_tflops, track_metrics
from .utils import is_torch_xla_available, is_torchao_available, setup_tf32


if is_torch_xla_available():
    from torch_xla import launch as xla_launch
    from torch_xla import step as xla_step
    from torch_xla import sync as xla_sync

if is_torchao_available():
    from .fp8 import FP8Manager


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
    micro_batch_size: int,
    sequence_length: int,
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

    fsdp_algorithm = None
    if hasattr(model, "set_requires_gradient_sync"):
        fsdp_algorithm = 2
    elif hasattr(model, "no_sync"):
        fsdp_algorithm = 1

    no_sync = nullcontext
    if not sync_every_gradient_accumulation_step:
        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        elif fsdp_algorithm == 2:
            model.set_requires_gradient_sync(False)

    metrics_tracker = MetricsTrackingDict({})
    optimizer_container.zero_grad()

    gradient_accumulation_steps = StepTracker.get_gradient_accumulation_steps()

    if tuning_method == TuningMethod.full_finetuning:
        # note the effect of gradient accumulation division is already in the lm_loss_multiplier
        batches = [get_next_batch(train_dataloader) for _ in range(gradient_accumulation_steps)]
        lm_loss_multiplier = gradient_accumulation_steps / sum([(batch["labels"] != -100).sum() for batch in batches])
    else:
        batches = None
        lm_loss_multiplier = 1 / (micro_batch_size * sequence_length)

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
    global_step_in_tokens: int,
    experiments_tracker: ExperimentsTracker,
    metrics_tracker: MetricsTrackingDict,
    group_name: str | None = None,
    context: str = "val",
) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        global_step_in_tokens (int): global step during training in number of tokens
        experiments_tracker (ExperimentsTracker): experiments tracker
        metrics_tracker (MetricsTrackingDict): metrics tracker
        group_name (str | None): group name for the validation / test set
        context (str): context
    """

    message = f"step = {global_step:,}, tokens = {global_step_in_tokens:,}"
    if group_name is not None:
        message += f", group_name = {group_name}"

    for key in metrics_tracker:
        if key == "tokens":
            continue

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

    tuning_method = args.tuning_args.tuning_method

    val_weighted_split_paths = args.datasets[0].class_args.get("val_weighted_split_paths")
    group_names = [None]
    if val_weighted_split_paths is not None:
        group_names = [key for key in val_weighted_split_paths.keys()[0]]

    model_container.train()
    micro_batch_size = args.training_parameters.micro_batch_size
    global_step = starting_iteration
    global_batch_size = StepTracker.get_global_batch_size()

    if tuning_method == TuningMethod.full_finetuning:
        train_dataloader_iterator = custom_iterator(train_dataloader, infinite=True)

        sequence_length = None
        tokens_per_batch = 0
        global_step_in_tokens = 0
    else:
        # train_dataloader is used for saving the state and we set it to None since we load using consumed_samples in
        # metadata during pretraining or distillation
        train_dataloader_iterator = train_dataloader
        train_dataloader = None

        sequence_length = args.datasets[0].class_args.get("sequence_length")
        tokens_per_batch = global_batch_size * sequence_length
        global_step_in_tokens = global_step * tokens_per_batch

    if eval_during_training:
        eval_steps = args.datasets[0].class_args.get("eval_steps")
        evaluate(
            val_dataloaders=val_dataloaders,
            model_container=model_container,
            global_step=global_step,
            global_step_in_tokens=global_step_in_tokens,
            experiments_tracker=experiments_tracker,
            eval_steps=eval_steps,
            group_names=group_names,
            tuning_method=tuning_method,
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
            context="val",
        )

    is_pipeline_parallel_enabled = args.distributed_args.num_pipeline_stages > 1
    if not is_pipeline_parallel_enabled:
        assert len(model_container) == 1

    if tuning_method == TuningMethod.full_finetuning:
        model_flops = None
    else:
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

    forward_context = loss_parallel if ProcessGroupManager.is_tensor_parallel_enabled() else nullcontext
    backward_context = loss_parallel if ProcessGroupManager.is_tensor_parallel_enabled() else nullcontext

    torch_profiler = TorchProfiler(args.logging_args.torch_profiler_trace_path)
    torch_profiler.__enter__()

    start_time = time.perf_counter()
    steps_since_start_time = 0
    metrics_tracker = MetricsTrackingDict({})

    while global_step < num_training_steps:
        global_step += 1
        steps_since_start_time += 1
        global_step_in_tokens += tokens_per_batch

        if is_pipeline_parallel_enabled:
            loss_step_dict = train_step_with_pipeline_parallel(
                model_container=model_container,
                pipeline_schedule=pipeline_schedule,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader_iterator,
                gradient_clipping=gradient_clipping,
                sequence_length=sequence_length,
            )
        else:
            loss_step_dict = train_step_without_pipeline_parallel(
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader_iterator,
                gradient_clipping=gradient_clipping,
                forward_context=forward_context,
                backward_context=backward_context,
                sync_every_gradient_accumulation_step=args.distributed_args.sync_every_gradient_accumulation_step,
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
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
            metrics_tracker["tokens"] = global_step_in_tokens

            track_metrics(
                global_step=global_step,
                global_step_in_tokens=global_step_in_tokens,
                experiments_tracker=experiments_tracker,
                metrics_tracker=metrics_tracker,
                context="train",
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0
            metrics_tracker = MetricsTrackingDict({})

        if eval_during_training and (global_step % eval_interval == 0 or global_step == num_training_steps):
            evaluate(
                val_dataloaders=val_dataloaders,
                model_container=model_container,
                global_step=global_step,
                global_step_in_tokens=global_step_in_tokens,
                experiments_tracker=experiments_tracker,
                eval_steps=eval_steps,
                group_names=group_names,
                tuning_method=tuning_method,
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                context="val",
            )

        if global_step % save_interval == 0 or global_step == num_training_steps:
            if tuning_method == TuningMethod.full_finetuning:
                metadata = {}
            else:
                metadata = {
                    "consumed_samples": global_step * global_batch_size,
                    "commit_id": Repo(Path(__file__).parents[1]).git.rev_parse("HEAD"),
                }

            save_checkpoint(
                args=args,
                model_container=model_container,
                optimizer_container=optimizer_container,
                lr_scheduler_container=lr_scheduler_container,
                train_dataloader=train_dataloader,
                experiments_tracker=experiments_tracker,
                iteration=global_step,
                metadata=metadata,
            )

            start_time = time.perf_counter()
            steps_since_start_time = 0

    if eval_during_training:
        if tuning_method == TuningMethod.full_finetuning:
            assert False
        else:
            evaluate(
                val_dataloaders=test_dataloaders,
                model_container=model_container,
                global_step=global_step,
                global_step_in_tokens=global_step_in_tokens,
                experiments_tracker=experiments_tracker,
                eval_steps=eval_steps,
                group_names=group_names,
                tuning_method=tuning_method,
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                context="test",
            )

    ensure_last_checkpoint_is_saved()
    torch_profiler.__exit__(None, None, None)


@torch.no_grad()
def evaluate(
    val_dataloaders: list[DataLoader],
    model_container: ModelContainer,
    global_step: int,
    global_step_in_tokens: int,
    experiments_tracker: ExperimentsTracker,
    eval_steps: int | None,
    group_names: list[str],
    tuning_method: TuningMethod,
    micro_batch_size: int,
    sequence_length: int,
    context: str,
) -> None:
    """main validation loop for the program

    Args:
        val_dataloaders (list[DataLoader]): list of validation dataloaders
        model_container (ModelContainer): container of models
        global_step (int): global step during training
        global_step_in_tokens (int): global step during training in number of tokens
        experiments_tracker (ExperimentsTracker): metrics tracker
        eval_steps (int): number of steps to run eval for
        group_names (list[str]): names of the datasets in validation/test group
        context (str): context
    """

    model_container.eval()

    assert len(model_container) == 1
    model = model_container[0]

    for group_name, val_dataloader in zip(group_names, val_dataloaders):
        is_val_dataloader_none = val_dataloader is None or len(val_dataloader) == 0

        if ProcessGroupManager.is_tensor_parallel_enabled():
            if not ProcessGroupManager.is_tensor_parallel_first_rank():
                is_val_dataloader_none = None

            is_val_dataloader_none = ProcessGroupManager.broadcast_object(
                is_val_dataloader_none,
                src=ProcessGroupManager.get_tensor_parallel_first_rank(),
                group=ProcessGroupManager.get_tensor_parallel_group(),
            )

        if is_val_dataloader_none:
            continue

        if eval_steps is None and tuning_method == TuningMethod.full_finetuning:
            eval_steps = torch.tensor(
                len(val_dataloader),
                device=Accelerator.get_current_device(),
                dtype=torch.int32 if Accelerator.get_accelerator() == Accelerator.trainium else torch.long,
            )

            torch.distributed.all_reduce(eval_steps, group=ProcessGroupManager.get_tensor_parallel_group())
            eval_steps = eval_steps.item()

        assert eval_steps is not None

        lm_loss_multiplier = 1 / eval_steps
        if tuning_method == TuningMethod.full_finetuning:
            val_dataloader = custom_iterator(val_dataloader, infinite=False)
        else:
            lm_loss_multiplier /= micro_batch_size * sequence_length

        metrics_tracker = MetricsTrackingDict({})
        loss_tokens = 0 if tuning_method == TuningMethod.full_finetuning else 1

        for _ in range(eval_steps):
            batch = get_next_batch(val_dataloader)
            if tuning_method == TuningMethod.full_finetuning:
                loss_tokens += (batch["labels"] != -100).sum()

            loss_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)
            metrics_tracker = metrics_tracker + loss_step_dict

        metrics_tracker = metrics_tracker / loss_tokens

        for key in metrics_tracker:
            metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)
        metrics_tracker["tokens"] = global_step_in_tokens

        track_val_metrics(
            global_step=global_step,
            global_step_in_tokens=global_step_in_tokens,
            experiments_tracker=experiments_tracker,
            metrics_tracker=metrics_tracker,
            group_name=group_name,
            context=context,
        )

    model_container.train()


def main(args_class: type[DistillationArgs | TrainingArgs] = TrainingArgs) -> None:
    """main program"""

    setup_tf32()

    args: TrainingArgs | DistillationArgs = get_args(args_class)
    tuning_method = args.tuning_args.tuning_method

    if args_class == TrainingArgs:
        assert tuning_method in [
            TuningMethod.pretraining,
            TuningMethod.full_finetuning,
        ], f"unexpected tuning method ({tuning_method})"
    elif args_class == DistillationArgs:
        assert args.distributed_args.fsdp_algorithm == 2, "Distillation is only supported with FSDP-2"

        assert tuning_method == TuningMethod.distillation, f"unexpected tuning method ({tuning_method})"

    # initialize distributed with nccl for multi-node communications
    process_group_manager = ProcessGroupManager(
        tensor_parallel_world_size=args.distributed_args.tensor_parallel_world_size,
        pipeline_parallel_world_size=args.distributed_args.pipeline_parallel_world_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        context_parallel_world_size=args.distributed_args.context_parallel_world_size,
        zero_stage=args.distributed_args.stage,
        timeout_minutes=args.distributed_args.timeout_minutes,
        use_async_tensor_parallel=args.distributed_args.use_async_tensor_parallel,
    )

    log_rank_0(logging.INFO, process_group_manager)
    log_rank_0(logging.INFO, f"total accelerators = {process_group_manager.get_world_size()}")
    log_rank_0(logging.INFO, f"tensor parallel size = {process_group_manager.get_tensor_parallel_world_size()}")
    log_rank_0(logging.INFO, f"pipeline parallel size = {process_group_manager.get_pipeline_parallel_world_size()}")
    log_rank_0(logging.INFO, f"data parallel size = {process_group_manager.get_data_parallel_world_size()}")
    log_rank_0(logging.INFO, f"context parallel size = {process_group_manager.get_context_parallel_world_size()}")

    args.log_args()
    log_environment()

    StepTracker(
        micro_batch_size=args.training_parameters.micro_batch_size,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
    )

    Accelerator.set_seed(args.random_args.seed)

    if tuning_method in [TuningMethod.distillation, TuningMethod.full_finetuning]:
        assert args.distributed_args.num_pipeline_stages == 1

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
        split_params_for_optimizer=args.optimizer_args.split_params_for_optimizer,
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
    metadata = {}
    train_dataloader = None
    tokenizer = model_container[0].tokenizer

    if tuning_method == TuningMethod.full_finetuning:
        train_dataloader = get_finetuning_dataloader(
            args, split=DatasetSplit.train, use_output=True, tokenizer=tokenizer
        )

        val_dataloaders = None
        if args.training_parameters.eval_during_training:
            val_dataloaders = [
                get_finetuning_dataloader(args, split=DatasetSplit.val, use_output=True, tokenizer=tokenizer)
            ]

        test_dataloaders = None

    if args.load_args is not None:
        starting_iteration, metadata, experiments_tracker_state_dict = load_checkpoint_for_training(
            args, args_class, model_container, optimizer_container, lr_scheduler_container, train_dataloader
        )

        if not args.load_args.load_dataloader_state:
            metadata["consumed_samples"] = 0

    if tuning_method != TuningMethod.full_finetuning:
        train_dataloader, val_dataloaders, test_dataloaders = get_pretraining_dataloaders(
            args, tokenizer, metadata.get("consumed_samples", 0)
        )

    experiments_tracker = ExperimentsTracker(
        experiments_tracker_name=args.logging_args.experiments_tracker_name,
        aim_args=args.logging_args.aim_args,
        wandb_args=args.logging_args.wandb_args,
        checkpoint_metadata=experiments_tracker_state_dict,
    )

    # track all hyperparams in args
    experiments_tracker.log_args(args, **model_container[0].calculate_num_parameters(return_dict=True))

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


def _xla_main(*args):
    main()


if __name__ == "__main__":
    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.tpu:
        xla_launch(_xla_main)
    else:
        main()
