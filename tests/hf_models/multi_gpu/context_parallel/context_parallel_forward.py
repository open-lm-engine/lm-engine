# **************************************************
# Copyright (c) 2026, Mayank Mishra, Shawn Tan
# **************************************************

import argparse
import os

import torch
import torch.distributed
from transformers import AutoModelForCausalLM

from lm_engine.accelerator import Accelerator
from lm_engine.enums import ContextParallelLoadBalancerMethod, Kernel
from lm_engine.hf_models import GPTBaseConfig, get_autoregressive_language_modeling_loss
from lm_engine.kernels import enable_kernels
from lm_engine.parallel import ProcessGroupManager, prepare_context_parallel_input
from lm_engine.parallel.context_parallel import _HeadTailLoadBalancer, _NoLoadBalancer
from lm_engine.utils import SafeTensorsWeightsManager, string_to_torch_dtype

from ....utils import from_config


parser = argparse.ArgumentParser()
parser.add_argument("--position-embedding-type", type=str)
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--dtype", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--load-balancing-method", type=ContextParallelLoadBalancerMethod, default=None)
args = parser.parse_args()

Accelerator.set_seed(42)

cp_world_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(
    context_parallel_world_size=cp_world_size,
    context_parallel_load_balancing_method=args.load_balancing_method,
)

dtype = string_to_torch_dtype(args.dtype)
num_key_value_heads = 8

config = GPTBaseConfig(
    num_layers=2,
    position_embedding_type=args.position_embedding_type,
    hidden_size=128,
    sequence_mixer_blocks=[
        {
            "sequence_mixer_type": "softmax_attention",
            "add_bias": False,
            "num_attention_heads": 16,
            "num_key_value_heads": num_key_value_heads,
        },
        {
            "sequence_mixer_type": "softmax_attention",
            "add_bias": False,
            "num_attention_heads": 16,
            "num_key_value_heads": num_key_value_heads,
        },
    ],
    mlp_blocks=[
        {"mlp_type": "MLP", "add_bias": False},
        {"mlp_type": "MLP", "add_bias": False},
    ],
)

kernels = []
if args.attention_implementation == "flash_attention_2":
    kernels.append(Kernel.flash_attention_2)
elif args.attention_implementation == "flash_attention_3":
    kernels.append(Kernel.flash_attention_3)
elif args.attention_implementation == "flash_attention_4":
    kernels.append(Kernel.flash_attention_4)

with enable_kernels(kernels):
    if torch.distributed.get_rank() == 0:
        with torch.device("meta"):
            model = from_config(config)

        model = model.to_empty(device=torch.cuda.current_device())
        for _, param in model.named_parameters():
            param.data.normal_(0, 0.0125)

        model.eval()
        model.save_pretrained(args.tmp_path, safe_serialization=True)
        model = model.to(dtype)

    ProcessGroupManager.barrier()

    # use dummy tensors to avoid initializing model here
    with torch.device("meta"):
        model_cp = AutoModelForCausalLM.from_config(config)

    # copy to device without copying storage
    model_cp = model_cp.to_empty(device=torch.cuda.current_device())

    # load weights into context parallel model — no weight sharding, each rank holds the full model
    model_cp.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(args.tmp_path))

    model_cp = model_cp.to(dtype)
    model_cp.eval()

    Accelerator.set_seed(42)

    batch_size = 4
    # must be divisible by cp_world_size * 2 for headtail load balancing
    sequence_length = 512

    # generate the same full-sequence inputs on every rank
    input_ids_full = torch.randint(
        0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
    )
    labels_full = torch.randint(
        0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
    )

    # _context_parallel_buffers (called inside prepare_context_parallel_input) reorders the tensors
    # in-place via buffer[i] = index_select(...). Clone before the call so the rank-0 reference
    # comparison still sees the original token order.
    input_ids_ref = input_ids_full.clone()
    labels_ref = labels_full.clone()

    # shard inputs across CP ranks — uses the load-balancing method from ProcessGroupManager
    input_ids_cp, labels_cp = prepare_context_parallel_input(inputs=(input_ids_full, labels_full))

    # ---- forward + backward with CP model ----
    output_cp = model_cp(input_ids=input_ids_cp)
    logits_cp = output_cp.logits[..., : config.vocab_size]
    logits_cp.retain_grad()

    loss_cp = get_autoregressive_language_modeling_loss(
        lm_logits=logits_cp,
        labels=labels_cp,
        shift_logits_and_labels=False,
        reduction="mean",
    )
    loss_cp.backward()

    # ---- gather logits and their gradients from all CP ranks ----
    cp_group = ProcessGroupManager.get_context_parallel_group()

    def _gather_and_restore(local: torch.Tensor, restore_idx: torch.Tensor) -> torch.Tensor:
        parts = [torch.zeros_like(local) for _ in range(cp_world_size)]
        torch.distributed.all_gather(parts, local.detach(), group=cp_group)
        gathered = torch.cat(parts, dim=1)  # [B, T, V] in sharded (reordered) order
        return gathered[:, restore_idx, :]

    # restore the original sequence order using the load-balancer's inverse permutation
    _LB_CLASSES = {None: _NoLoadBalancer, ContextParallelLoadBalancerMethod.headtail: _HeadTailLoadBalancer}
    lb = _LB_CLASSES[args.load_balancing_method](sequence_length, cp_world_size, torch.cuda.current_device())
    restore_indices = lb._generate_indices(restore=True).squeeze(0).long()

    logits_cp_full = _gather_and_restore(logits_cp, restore_indices)
    grad_logits_cp_full = _gather_and_restore(logits_cp.grad, restore_indices)

    if torch.distributed.get_rank() == 0:
        # run full-sequence model on rank 0 without CP active;
        # forward, loss, and backward must all be inside the context manager to avoid
        # any DTensor CP operations (e.g. CP-aware position-id sharding or loss reduction)
        with ProcessGroupManager.set_dummy_context_parallel_world_size(1):
            output = model(input_ids=input_ids_ref)
            logits = output.logits[..., : config.vocab_size]
            logits.retain_grad()

            loss = get_autoregressive_language_modeling_loss(
                lm_logits=logits,
                labels=labels_ref,
                shift_logits_and_labels=False,
                reduction="mean",
            )
            loss.backward()

        error = (logits - logits_cp_full).abs().max()
        assert error < 5e-4, f"logits don't match for normal and context parallel model, error is ({error})"

        error = (loss - loss_cp).abs().max()
        assert error < 1e-3, f"losses don't match for normal and context parallel model, error is ({error})"

        error = (logits.grad - grad_logits_cp_full).abs().max()
        assert error < 5e-4, f"logit gradients don't match for normal and context parallel model, error is ({error})"
