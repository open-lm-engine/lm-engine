# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed._composable.fsdp import MixedPrecisionPolicy as MixedPrecision2
from torch.distributed._composable.fsdp import OffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as MixedPrecision1
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from .arguments import TrainingArgs
from .containers import ModelContainer
from .enums import Kernel
from .gradient_checkpointing import apply_gradient_checkpointing
from .hf_models import CausalLMOutputWithPast
from .hf_models.parameter import _ALL_MARKERS
from .kernels import is_kernel_allowed
from .utils import (
    ProcessGroupManager,
    get_module_class_from_name,
    is_torchao_available,
    log_rank_0,
    string_to_torch_dtype,
)


if is_torchao_available():
    from torchao.float8 import ScalingType

    from .fp8 import FP8Manager

torch._inductor.config.reorder_for_compute_comm_overlap = True


_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}


def _get_pipeline_parallel_schedule(
    pipeline_parallel_schedule: str,
    gradient_accumulation_steps: int,
    pipeline_stages: list[PipelineStage],
    loss_fn: Callable,
) -> _PipelineSchedule:
    try:
        schedule_class = get_schedule_class(pipeline_parallel_schedule)
    except ValueError:
        raise ValueError(
            f"unexpected schedule ({pipeline_parallel_schedule}), expected values are: ['1F1B', "
            "'Interleaved1F1B', 'GPipe', 'FlexibleInterleaved1F1B', 'LoopedBFS', 'InterleavedZeroBubble', "
            "'PipelineScheduleSingle', 'PipelineScheduleMulti']"
        )

    if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
        raise NotImplementedError()

    if issubclass(schedule_class, PipelineScheduleSingle):
        assert len(pipeline_stages) == 1

    def custom_loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dict = loss_fn(output, target)
        return loss_dict["loss"]

    return schedule_class(
        pipeline_stages if issubclass(schedule_class, PipelineScheduleMulti) else pipeline_stages[0],
        n_microbatches=gradient_accumulation_steps,
        loss_fn=custom_loss_function,
    )


def _get_fsdp_mixed_precision(
    dtype: torch.dtype, communication_dtype: torch.dtype | None, fsdp_algorithm: int
) -> MixedPrecision1:
    if communication_dtype is None:
        communication_dtype = dtype

    if fsdp_algorithm == 1:
        mixed_precision = MixedPrecision1(param_dtype=dtype, reduce_dtype=communication_dtype, buffer_dtype=dtype)
    else:
        mixed_precision = MixedPrecision2(param_dtype=dtype, reduce_dtype=communication_dtype)

    return mixed_precision


def _get_parameter_marker_maps(model_container: ModelContainer) -> list[dict]:
    marker_maps = []
    for model in model_container:
        marker_maps.append({})
        for param_name, param in model.named_parameters():
            marker_maps[-1][param_name] = {}
            for marker in _ALL_MARKERS:
                marker_maps[-1][param_name][marker] = getattr(param, marker, False)

    return marker_maps


def _set_parameter_marker_maps(model_container: ModelContainer, marker_maps: list[dict]) -> None:
    for original_param_name in marker_maps[0]:
        break

    for new_param_name, _ in model_container[0].named_parameters():
        break

    prefix = new_param_name.split(original_param_name)[0]

    for model, _marker_map in zip(model_container, marker_maps):
        for new_param_name, parameter in model.named_parameters():
            original_param_name = new_param_name.split(prefix)[-1]

            for marker, value in _marker_map[original_param_name].items():
                setattr(parameter, marker, value)


def wrap_model_container_for_distributed_training(
    args: TrainingArgs, model_container: ModelContainer
) -> tuple[ModelContainer, _PipelineSchedule]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model_container (ModelContainer): model container

    Returns:
        tuple[ModelContainer, _PipelineSchedule]: container of parallelized models and pipeline schedule
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload
    torch_compile = args.distributed_args.torch_compile
    dtype = args.mixed_precision_args.dtype
    communication_dtype = args.distributed_args.communication_dtype
    efficient_initialization = args.model_args.efficient_initialization
    fsdp_algorithm = args.distributed_args.fsdp_algorithm
    num_pipeline_stages = args.distributed_args.num_pipeline_stages
    data_parallel_sharding_world_size = ProcessGroupManager.get_data_parallel_sharding_world_size()
    data_parallel_replication_world_size = ProcessGroupManager.get_data_parallel_replication_world_size()
    model_name = args.model_args.model_name

    if dtype in ["fp16", "bf16"]:
        if communication_dtype != "fp32":
            log_rank_0(
                logging.WARN,
                f"using ({communication_dtype}) with mixed precision training in ({dtype}), recommended is to use ({torch.float32})",
            )
    elif dtype == "fp8":
        assert is_torchao_available(), "torchao is needed for FP8 training"

        FP8Manager(
            model_container,
            enable_fsdp_fp8_all_gather=ProcessGroupManager.get_data_parallel_sharding_world_size() > 1,
            precompute_fp8_dynamic_scale_for_fsdp=True,
            torch_compile=torch_compile,
            scaling_type_input=ScalingType(args.mixed_precision_args.scaling_type_input),
            scaling_type_weight=ScalingType(args.mixed_precision_args.scaling_type_weight),
            scaling_type_grad_output=ScalingType(args.mixed_precision_args.scaling_type_grad_output),
        )

        dtype = "bf16"

    block_names = model_container[0].model._no_split_modules
    teacher_block_names = (
        model_container[0].teacher_model._no_split_modules if model_container[0].has_teacher_model() else []
    )

    dtype = None if dtype is None else string_to_torch_dtype(dtype)
    communication_dtype = None if communication_dtype is None else string_to_torch_dtype(communication_dtype)

    assert stage in [0, 2, 3]

    dp_mesh = ProcessGroupManager.get_data_parallel_mesh()
    block_classes = [
        get_module_class_from_name(model_container[0], name) for name in block_names + teacher_block_names
    ]

    marker_maps = _get_parameter_marker_maps(model_container)

    if args.distributed_args.gradient_checkpointing_method is not None:
        assert len(block_names) == 1

        for model in model_container:
            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )

    # for PP, we use FSDP-2 always
    use_ddp = (stage == 0 or data_parallel_sharding_world_size == 1) and num_pipeline_stages == 1

    mixed_precision_policy = _get_fsdp_mixed_precision(
        dtype=dtype,
        communication_dtype=communication_dtype,
        fsdp_algorithm=2 if use_ddp else fsdp_algorithm,
    )

    if use_ddp or fsdp_algorithm == 2:
        log_rank_0(logging.INFO, "using FSDP-2")
        zero3 = stage == 3

        def _sharding_function(parameter: nn.Parameter) -> Shard:
            dps = (
                ProcessGroupManager.get_data_parallel_world_size()
                if data_parallel_sharding_world_size is None
                else data_parallel_sharding_world_size
            )

            if parameter.size(0) > dps or parameter.dim() == 1:
                return Shard(0)
            else:
                for dim in range(1, parameter.dim()):
                    if parameter.size(dim) > dps and parameter.size(dim) % dps == 0:
                        return Shard(dim)

                log_rank_0(logging.WARN, "sharding along dim=0 since no suitable sharding dimension was found")
                return Shard(0)

        for i, model in enumerate(model_container):
            if efficient_initialization and model_name is not None:
                # state dict with Tensors
                old_state_dict = model.state_dict()

            for module in model.modules():
                if isinstance(module, tuple(block_classes)):
                    fully_shard(
                        module,
                        mesh=dp_mesh,
                        reshard_after_forward=zero3,
                        shard_placement_fn=_sharding_function,
                        mp_policy=mixed_precision_policy,
                        offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
                    )

            fully_shard(
                model,
                mesh=dp_mesh,
                reshard_after_forward=zero3,
                shard_placement_fn=None if use_ddp else _sharding_function,
                mp_policy=mixed_precision_policy,
                offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
            )

            if efficient_initialization:
                # contributed by Yu Chin Fabian Lim
                # original reference https://github.com/fabianlim/accelerate/pull/1
                if model_name is None:
                    model = model.to_empty(device=torch.cuda.current_device())

                    for module in model.modules():
                        if hasattr(module, "reset_parameters"):
                            with torch.device(torch.cuda.current_device()):
                                module.reset_parameters()
                else:
                    if ProcessGroupManager.get_data_parallel_rank() == 0:
                        model = model.to(torch.cuda.current_device())
                    else:
                        model = model.to_empty(device=torch.cuda.current_device())

                        for module in model.modules():
                            if hasattr(module, "reset_parameters"):
                                with torch.device(torch.cuda.current_device()):
                                    module.reset_parameters()

                    # state dict with DTensors
                    new_state_dict = model.state_dict()

                    for param_name, param in old_state_dict.items():
                        if ProcessGroupManager.get_data_parallel_rank() == 0:
                            full_tensor = param
                        else:
                            full_tensor = torch.empty(
                                param.shape, dtype=param.dtype, device=torch.cuda.current_device()
                            )

                        new_state_dict[param_name] = distribute_tensor(
                            full_tensor,
                            device_mesh=new_state_dict[param_name].device_mesh,
                            placements=new_state_dict[param_name].placements,
                        )

                    model.load_state_dict(new_state_dict, assign=True)
                    del old_state_dict, new_state_dict
    elif fsdp_algorithm == 1:
        log_rank_0(logging.INFO, "using FSDP-1")
        assert num_pipeline_stages == 1

        sharding_strategy = (
            _STAGE_FULL_SHARDING_STRATEGY_MAP[stage]
            if data_parallel_replication_world_size == 1
            else _STAGE_HYBRID_SHARDING_STRATEGY_MAP[stage]
        )

        def _param_init(module: nn.Module) -> None:
            assert len(teacher_block_names) == 0, "efficient initialization doesn't support distillation"

            if model_name is None:
                module = module.to_empty(device=torch.cuda.current_device())

                if hasattr(module, "reset_parameters"):
                    with torch.no_grad():
                        module.reset_parameters()
            else:
                if ProcessGroupManager.get_data_parallel_rank() != 0:
                    module = module.to_empty(device=torch.cuda.current_device())

        for i, model in enumerate(model_container):
            model_container[i] = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
                mixed_precision=mixed_precision_policy,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=block_classes),
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                use_orig_params=True,
                # https://github.com/meta-llama/llama-recipes/blob/492455dc080f6c25f356e283e443be0cce86aaeb/src/llama_recipes/finetuning.py#L191
                sync_module_states=efficient_initialization,
                param_init_fn=_param_init if efficient_initialization else None,
                device_mesh=dp_mesh,
            )
    else:
        raise ValueError(f"unexpected fsdp_algorithm ({fsdp_algorithm})")

    if torch_compile:
        log_rank_0(logging.INFO, "using torch compile")

        for i, model in enumerate(model_container):
            model_container[i] = torch.compile(model)

    _set_parameter_marker_maps(model_container, marker_maps)

    pipeline_stages = []
    pipeline_schedule = None

    if num_pipeline_stages > 1:
        micro_batch_size = args.training_parameters.micro_batch_size
        sequence_length = args.datasets[0].class_args.get("sequence_length")

        pipeline_parallel_schedule = args.distributed_args.pipeline_parallel_schedule
        gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps

        if pipeline_parallel_schedule == "1F1B":
            assert (
                gradient_accumulation_steps % num_pipeline_stages == 0
            ), f"gradient_accumulation_steps ({gradient_accumulation_steps}) should be divisible by num_pipeline_stages ({num_pipeline_stages})"

        for model in model_container:
            intermediate_dtype = string_to_torch_dtype(args.mixed_precision_args.dtype)

            dummy_input_tensor = model.model.get_dummy_input_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )
            dummy_output_tensor = model.model.get_dummy_output_tensor(
                micro_batch_size,
                sequence_length,
                intermediate_dtype=intermediate_dtype,
                output_parallel_lm_logits_if_possible=True,
            )

            stage = PipelineStage(
                model,
                stage_index=model.pipeline_stage_id,
                num_stages=num_pipeline_stages,
                device=torch.cuda.current_device(),
                input_args=dummy_input_tensor,
                output_args=dummy_output_tensor,
                group=ProcessGroupManager.get_pipeline_parallel_group(),
            )

            pipeline_stages.append(stage)

        lm_loss_multiplier = 1 / (
            args.training_parameters.micro_batch_size * args.datasets[0].class_args.get("sequence_length")
        )

        def _pipeline_parallel_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            use_fused_linear_cross_entropy = is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

            if isinstance(input, tuple):
                input, aux_loss = input
            else:
                aux_loss = 0

            output = CausalLMOutputWithPast(
                logits=None if use_fused_linear_cross_entropy else input,
                aux_loss=aux_loss,
                last_hidden_state=input if use_fused_linear_cross_entropy else None,
            )
            loss = model.get_loss(output, target, lm_loss_multiplier)

            return loss

        pipeline_schedule = _get_pipeline_parallel_schedule(
            pipeline_parallel_schedule=args.distributed_args.pipeline_parallel_schedule,
            gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
            pipeline_stages=pipeline_stages,
            loss_fn=_pipeline_parallel_loss,
        )

    return model_container, pipeline_schedule
