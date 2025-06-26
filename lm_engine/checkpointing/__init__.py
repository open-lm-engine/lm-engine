# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import logging
import os
import random

import numpy as np
import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from ..arguments import InferenceArgs, TrainingArgs, UnshardingArgs
from ..containers import LRSchedulerContainer, ModelContainer, OptimizerContainer
from ..data import ResumableDataLoader
from ..enums import Mode
from ..hf_models import fix_unsharded_state_dict
from ..model_wrapper import ModelWrapper, get_model_container
from ..utils import ExperimentsTracker, ProcessGroupManager, load_yaml, log_rank_0, run_rank_n, string_to_torch_dtype
from .lr_scheduler import _get_lr_scheduler_path, _LRSchedulerSaver, _resume_learning_rate
from .model import _get_model_path, _ModelSaver
from .model_optimizer import _get_model_optimizer_path, _ModelOptimizerSaver
from .optimizer import _get_optimizer_path, _OptimizerSaver


_TRAINING_CONFIG_PREFIX = "training_config"
_INFERENCE_CONFIG_PREFIX = "inference_config"
_KILLSWITCH = "KILLSWITCH"
_FUTURE = None


def ensure_last_checkpoint_is_saved() -> None:
    if _FUTURE is not None:
        _FUTURE.result()


def save_checkpoint(
    args: TrainingArgs,
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer | None,
    lr_scheduler_container: LRSchedulerContainer | None,
    train_dataloader: ResumableDataLoader,
    experiments_tracker: ExperimentsTracker,
    iteration: int,
    metadata: dict | None = None,
) -> None:
    """save checkpoint during training

    Args:
        args (TrainingArgs): arguments for training
        model_container (ModelContainer): models to save
        optimizer_container (OptimizerContainer): optimizers to save
        lr_scheduler_container (LRSchedulerContainer): learning rate schedulers to save
        train_dataloader (DataLoader): train dataloader to save
        experiments_tracker (ExperimentsTracker): experiment tracker to save
        iteration (int): current iteration
        metadata (dict | None): extra stuff to store

    Raises:
        ValueError: if unexpected distributed backend is found
    """

    save_path = _get_base_path(args.save_args.save_path, iteration)
    os.makedirs(save_path, exist_ok=True)

    ensure_last_checkpoint_is_saved()

    if lr_scheduler_container is None:
        log_rank_0(
            logging.WARN,
            "lr_scheduler_container is not passed to save_checkpoint. Therefore, the function will not save the lr_scheduler",
        )
    else:
        lr_scheduler_path = _get_lr_scheduler_path(save_path)
        os.makedirs(os.path.dirname(lr_scheduler_path), exist_ok=True)
        torch.save(_LRSchedulerSaver(lr_scheduler_container).state_dict(), _get_lr_scheduler_path(save_path))

    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    rng_state_path = _get_rng_state_path(save_path)
    os.makedirs(os.path.dirname(rng_state_path), exist_ok=True)
    torch.save(rng_state, rng_state_path)

    if train_dataloader is not None:
        dataloader_path = _get_dataloader_path(save_path)
        os.makedirs(os.path.dirname(dataloader_path), exist_ok=True)
        torch.save(train_dataloader.state_dict(), dataloader_path)

    if experiments_tracker is not None:
        run_rank_n(json.dump)(
            experiments_tracker.state_dict(), run_rank_n(open)(_get_experiments_tracker_path(save_path), "w"), indent=4
        )

    if metadata is not None:
        run_rank_n(json.dump)(metadata, run_rank_n(open)(_get_metadata_path(save_path), "w"), indent=4)

    save_args(args, save_path, mode=Mode.training)

    if args.save_args.save_optimizer:
        if optimizer_container is None:
            log_rank_0(
                logging.WARN,
                "optimizer_container is not passed to save_checkpoint but save_optimizer is set to True. "
                "Therefore, the function will not save the optimizer",
            )

    if args.save_args.async_checkpointing:
        global _FUTURE
        _FUTURE = dcp.async_save(
            {
                "state": _ModelOptimizerSaver(
                    model_container=model_container,
                    optimizer_container=optimizer_container if args.save_args.save_optimizer else None,
                )
            },
            checkpoint_id=_get_model_optimizer_path(save_path),
        )

        def _f(_future):
            run_rank_n(json.dump)(
                {"latest_checkpointed_iteration": iteration},
                run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
                indent=4,
            )

            log_rank_0(logging.INFO, f"checkpoint saved asynchronously at {iteration}")

            if os.path.exists(os.path.join(args.save_args.save_path, _KILLSWITCH)):
                exit()

        _FUTURE.add_done_callback(_f)
    else:
        dcp.save({"state": _ModelSaver(model_container)}, checkpoint_id=_get_model_path(save_path))

        if args.save_args.save_optimizer and optimizer_container is not None:
            dcp.save(
                {"state": _OptimizerSaver(model_container, optimizer_container)},
                checkpoint_id=_get_optimizer_path(save_path),
            )

        torch.distributed.barrier()

        run_rank_n(json.dump)(
            {"latest_checkpointed_iteration": iteration},
            run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
            indent=4,
        )

        log_rank_0(logging.INFO, f"checkpoint saved at {iteration}")

        if os.path.exists(os.path.join(args.save_args.save_path, _KILLSWITCH)):
            ProcessGroupManager.destroy_process_groups()
            exit()


def load_checkpoint_for_training(
    args: TrainingArgs,
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
) -> tuple[int, dict, dict]:
    """load checkpoint for training

    Args:
        args (TrainingArgs): arguments for training
        model_container (ModelContainer): models to save
        optimizer_container (OptimizerContainer): optimizers to save
        lr_scheduler_container (LRSchedulerContainer): learning rate schedulers to save
        train_dataloader (ResumableDataLoader): train dataloader to load

    Raises:
        ValueError: if unexpected distributed backend is found

    Returns:
        tuple[int, dict, dict]: checkpointed iteration, metadata, experiments_tracker state dict
    """

    if args.load_args is None or args.load_args.load_path is None:
        return

    load_optimizer = args.load_args.load_optimizer
    load_rng_state = args.load_args.load_rng_state
    load_dataloader_state = args.load_args.load_dataloader_state
    load_experiments_tracker_state = args.load_args.load_experiments_tracker_state
    load_starting_iteration = args.load_args.load_starting_iteration

    iteration = args.load_args.iteration
    if iteration is None:
        iteration = json.load(open(_get_latest_checkpointed_iterations_path(args.load_args.load_path), "r"))[
            "latest_checkpointed_iteration"
        ]

    load_path = _get_base_path(args.load_args.load_path, iteration)

    args_file = os.path.join(load_path, f"{_TRAINING_CONFIG_PREFIX}.yml")
    args_from_checkpoint = load_yaml(args_file)
    args_from_checkpoint = TrainingArgs(**args_from_checkpoint)

    log_rank_0(logging.INFO, f"loading checkpoint saved at {load_path}")

    if args_from_checkpoint.save_args.async_checkpointing:
        saver = _ModelOptimizerSaver(model_container, optimizer_container if load_optimizer else None)
        state_dict = {"state": saver.state_dict()}
        dcp.load(state_dict, checkpoint_id=_get_model_optimizer_path(load_path))
        saver.load_state_dict(state_dict["state"])
    else:
        saver = _ModelSaver(model_container)
        state_dict = {"state": saver.state_dict()}
        dcp.load(state_dict, checkpoint_id=_get_model_path(load_path))
        saver.load_state_dict(state_dict["state"])

        if load_optimizer:
            saver = _OptimizerSaver(model_container, optimizer_container)
            state_dict = {"state": saver.state_dict()}
            dcp.load(state_dict, checkpoint_id=_get_optimizer_path(load_path))
            saver.load_state_dict(state_dict["state"])

    del saver, state_dict

    if args.load_args.load_lr_scheduler:
        assert load_optimizer, "load_lr_scheduler requires loading of optimizer"

        _LRSchedulerSaver(lr_scheduler_container).load_state_dict(
            torch.load(_get_lr_scheduler_path(load_path), weights_only=False)
        )
    elif args.load_args.resume_learning_rate:
        for model, optimizer, lr_scheduler in zip(model_container, optimizer_container, lr_scheduler_container):
            _resume_learning_rate(
                args,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                iteration=iteration if load_starting_iteration else None,
            )

    if load_rng_state:
        rng_state = torch.load(_get_rng_state_path(load_path), weights_only=False)
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])

    metadata = None
    if os.path.isfile(_get_metadata_path(load_path)):
        metadata = json.load(open(_get_metadata_path(load_path), "r"))

    if load_dataloader_state and train_dataloader is not None:
        train_dataloader.load_state_dict(torch.load(_get_dataloader_path(load_path), weights_only=False))

    experiments_tracker_json = None
    if load_experiments_tracker_state and os.path.exists(_get_experiments_tracker_path(load_path)):
        experiments_tracker_json = json.load(open(_get_experiments_tracker_path(load_path), "r"))

    if not load_starting_iteration:
        iteration = 0

    return iteration, metadata, experiments_tracker_json


def load_checkpoint_for_inference(
    args: InferenceArgs | UnshardingArgs, mode: Mode, allowed_meta_device: bool = False
) -> tuple[ModelWrapper, TrainingArgs, dict]:
    """load checkpoint for inference

    Args:
        args (Union[InferenceArgs, UnshardingArgs]): arguments
        mode (Mode): training/inference mode
        allowed_meta_device (bool): whether to use meta device
    """

    load_path = args.load_args.load_path

    iteration = args.load_args.iteration
    if iteration is None:
        iteration = json.load(open(_get_latest_checkpointed_iterations_path(args.load_args.load_path), "r"))[
            "latest_checkpointed_iteration"
        ]

    log_rank_0(logging.INFO, f"loading checkpoint saved at {_get_base_path(load_path, iteration)}")

    args_file = os.path.join(_get_base_path(load_path, iteration), f"{_TRAINING_CONFIG_PREFIX}.yml")
    args_from_checkpoint = load_yaml(args_file)

    if "teacher_args" in args_from_checkpoint:
        args_from_checkpoint["tuning_args"]["tuning_method"] = "pretraining"
        args_from_checkpoint.pop("teacher_args")

    args_from_checkpoint = TrainingArgs(**args_from_checkpoint)

    if args.mixed_precision_args is not None:
        log_rank_0(logging.INFO, "overriding mixed precision args")
        args_from_checkpoint.mixed_precision_args = args.mixed_precision_args

    checkpoint_tp_world_size = args_from_checkpoint.distributed_args.tensor_parallel_world_size
    use_meta = args_from_checkpoint.model_args.model_name is None if allowed_meta_device else False

    with (
        torch.device("meta") if use_meta else torch.device(torch.cuda.current_device()),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
    ):
        original_num_stages = args_from_checkpoint.distributed_args.num_pipeline_stages
        args_from_checkpoint.distributed_args.num_pipeline_stages = 1

        model = get_model_container(args_from_checkpoint, mode)[0]

        args_from_checkpoint.distributed_args.num_pipeline_stages = original_num_stages

    if use_meta:
        model = model.to_empty(device="cpu")

    state = {}
    if args_from_checkpoint.save_args.async_checkpointing:
        _load_state_dict(
            state,
            storage_reader=FileSystemReader(_get_model_optimizer_path(_get_base_path(load_path, iteration))),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )

        state = state["state"]["model"]
    else:
        _load_state_dict(
            state,
            storage_reader=FileSystemReader(_get_model_path(_get_base_path(load_path, iteration))),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )

        state = state["state"]

    if checkpoint_tp_world_size > 1:
        state = fix_unsharded_state_dict(
            model.config, state, tensor_parallel_world_size=checkpoint_tp_world_size, prefix="model."
        )

    dtype = string_to_torch_dtype(model.dtype)
    for key in list(state.keys()):
        state[key] = state[key].to(dtype)

    model.load_state_dict(state)

    return model, args_from_checkpoint, state


@run_rank_n
def save_args(args: TrainingArgs | InferenceArgs, save_path: str, mode: Mode) -> None:
    """saves training args as a json

    Args:
        args (TrainingArgs | InferenceArgs): arguments for training or inference
        save_path (str): save location on disk
    """

    file_prefix = _TRAINING_CONFIG_PREFIX if mode == Mode.training else _INFERENCE_CONFIG_PREFIX
    save_path = os.path.join(save_path, f"{file_prefix}.yml")
    yaml.dump(args.to_dict(), open(save_path, "w"), indent=2)


def _get_checkpoint_tag(iteration: int) -> str:
    return f"global_step{iteration}"


def _get_base_path(path: str, iteration: int) -> str:
    return os.path.join(path, _get_checkpoint_tag(iteration))


def _get_dataloader_path(path: str) -> str:
    return os.path.join(path, "dataloader", f"dataloader-{ProcessGroupManager.get_data_parallel_rank()}.pt")


def _get_rng_state_path(path: str) -> str:
    return os.path.join(path, "rng_state", f"rng_state-{ProcessGroupManager.get_global_rank()}.pt")


def _get_latest_checkpointed_iterations_path(path: str) -> str:
    return os.path.join(path, "latest_checkpointed_iteration.json")


def _get_experiments_tracker_path(path: str) -> str:
    return os.path.join(path, "experiments_tracker.json")


def _get_metadata_path(path: str) -> str:
    return os.path.join(path, "metadata.json")
