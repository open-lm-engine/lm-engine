# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch
from torch.distributed import ReduceOp
from transformers import AutoConfig

from .enums import GradientCheckpointingMethod
from .hf_models import CommonConfig, is_custom_model
from .hf_models.modeling_utils import is_glu
from .utils import ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager, divide_if_divisible, log_metrics


def all_reduce_metrics_tracker(metrics_tracker: MetricsTrackingDict) -> MetricsTrackingDict:
    tensor = [metrics_tracker[key] for key in metrics_tracker]
    tensor = torch.stack(tensor)
    # NOTE the cpu() call was to save memory but might not be needed anymore
    # tensor = torch.stack(tensor) / ProcessGroupManager.get_data_parallel_world_size()
    # tensor = tensor.cpu()
    # gloo op doesn't support averaging so we do sum and divide by world size above
    torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())
    tensor = tensor.tolist()

    for i, key in enumerate(metrics_tracker):
        metrics_tracker[key] = tensor[i]

    return metrics_tracker


def track_metrics(
    global_step: int, experiments_tracker: ExperimentsTracker, metrics_tracker: MetricsTrackingDict, context: str
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        metrics_tracker (float): metrics tracker
        context (str): experiment context
    """

    # experiments tracker
    experiments_tracker.track(metrics_tracker.get_dict(), step=global_step, context=context)

    message = f"step = {global_step}"
    for key in metrics_tracker:
        if key == "learning_rate":
            message += f", {key} = {metrics_tracker[key]:.4e}"
        else:
            message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_metrics(logging.INFO, message)


def get_torch_profiler(torch_profiler_trace_path: str) -> torch.profiler.profile:
    torch_profiler = None
    if torch_profiler_trace_path is not None:
        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_trace_path),
            record_shapes=True,
        )

    return torch_profiler


def _get_linear_flops(m: int, k: int, n: int, gradient_checkpointing: bool = False) -> int:
    forward_flops = 2 * m * k * n
    backward_flops = 2 * forward_flops

    total_flops = forward_flops + backward_flops
    if gradient_checkpointing:
        total_flops += forward_flops

    return total_flops


def _get_attention_flops(batch_size: int, sequence_length: int, hidden_size: int) -> int:
    attention_forward_flops = 2 * batch_size * sequence_length * (sequence_length + 1) * hidden_size
    attention_backward_flops = 5 * attention_forward_flops / 2
    return attention_forward_flops + attention_backward_flops


def get_model_tflops(
    config: AutoConfig | CommonConfig,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing_method: GradientCheckpointingMethod | None,
    gradient_checkpointing_args: dict,
) -> None:
    if not is_custom_model(config.model_type):
        return 0

    b = batch_size
    s = sequence_length
    h = config.hidden_size
    v = config.vocab_size

    num_layers_checkpointed = (
        gradient_checkpointing_args.get("num_blocks", config.num_layers)
        if gradient_checkpointing_method == GradientCheckpointingMethod.block
        else 0
    )

    total_flops = 0
    for layer_idx in range(config.num_layers):
        block = config.sequence_mixer_blocks[layer_idx]
        sequence_mixer_type = block.sequence_mixer_type
        gradient_checkpointing_enabled = layer_idx < num_layers_checkpointed

        if sequence_mixer_type == "causal_convolution":
            sequence_mixer_flops = _get_linear_flops(
                b * s, h, block.in_channels, gradient_checkpointing=gradient_checkpointing_enabled
            )
            sequence_mixer_flops += divide_if_divisible(
                _get_linear_flops(
                    b * s, block.in_channels, block.out_channels, gradient_checkpointing=gradient_checkpointing_enabled
                ),
                block.num_groups,
                "",
            )
            sequence_mixer_flops += _get_linear_flops(
                b * s, block.out_channels, h, gradient_checkpointing=gradient_checkpointing_enabled
            )
        elif sequence_mixer_type in ["softmax_attention", "stickbreaking_attention"]:
            # QKV projection FLOPs
            sequence_mixer_flops = _get_linear_flops(
                b * s,
                h,
                h * (1 + 2 * block.num_key_value_heads / block.num_attention_heads),
                gradient_checkpointing=gradient_checkpointing_enabled,
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(
                b * s, h, h, gradient_checkpointing=gradient_checkpointing_enabled
            )

            sequence_mixer_flops += _get_attention_flops(b, s, h)
        elif sequence_mixer_type == "multihead_latent_attention":
            # QKV down and up projection FLOPs
            sequence_mixer_flops = 2 * _get_linear_flops(
                b * s,
                h,
                block.query_compression_size + 2 * block.key_value_compression_size,
                gradient_checkpointing=gradient_checkpointing_enabled,
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(
                b * s, h, h, gradient_checkpointing=gradient_checkpointing_enabled
            )

            sequence_mixer_flops += _get_attention_flops(b, s, h)
        elif sequence_mixer_type == "mamba2":
            # NOTE taken from NexaAI's fork (might be incorrect)
            # Mamba2 FLOP calculation based on its specific architecture
            # Core components: projection, convolution, SSM operations
            # Input projection + convolution + SSM computation + output projection
            # TODO fix this for gradient checkpointing
            projection_flops = 4 * b * s * h * block.intermediate_size
            ssm_flops = 4 * b * s * block.intermediate_size * block.state_size

            sequence_mixer_flops = projection_flops + ssm_flops
            sequence_mixer_flops *= 2
        elif sequence_mixer_type == "rnn":
            # input projection FLOPs
            sequence_mixer_flops = _get_linear_flops(b * s, h, block.state_size)
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(b * s, block.state_size, h)

            head_dim = block.state_size / block.num_heads

            # sigmoid(Wh + x)
            sequence_mixer_flops += s * block.num_heads * (_get_linear_flops(b, head_dim, head_dim) + b * head_dim)
        elif sequence_mixer_type == "gru":
            # input projection FLOPs
            sequence_mixer_flops = _get_linear_flops(b * s, h, 3 * block.state_size)
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(b * s, block.state_size, h)

            head_dim = block.state_size / block.num_heads

            # sigmoid(Wh + x)
            sequence_mixer_flops += 3 * s * block.num_heads * (_get_linear_flops(b, head_dim, head_dim) + b * head_dim)
        else:
            raise NotImplementedError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")

        total_flops += sequence_mixer_flops

        block = config.mlp_blocks[layer_idx]

        # 2x for input and output linear layer
        mlp_flops = 2 * _get_linear_flops(
            b * s, h, block.intermediate_size, gradient_checkpointing=gradient_checkpointing_enabled
        )
        if block.mlp_type == "MoE":
            mlp_flops *= block.num_experts_per_tok

        if is_glu(block.activation_function):
            mlp_flops *= 1.5

        total_flops += mlp_flops

    total_flops += _get_linear_flops(b * s, h, v)
    total_flops /= 10**12

    return total_flops
