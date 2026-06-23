# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed
from transformers import AutoModelForCausalLM

from lm_engine.accelerator import Accelerator
from lm_engine.enums import Kernel
from lm_engine.hf_models import GPTBaseConfig
from lm_engine.kernels import enable_kernels
from lm_engine.parallel import ProcessGroupManager
from lm_engine.utils import SafeTensorsWeightsManager, string_to_torch_dtype

from ....utils import from_config


parser = argparse.ArgumentParser()
parser.add_argument("--position-embedding-type", type=str)
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--dtype", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--use-padding-free-transformer", action="store_true")
parser.add_argument("--sequence-parallel", action="store_true")
args = parser.parse_args()

Accelerator.set_seed(42)

ProcessGroupManager(tensor_parallel_world_size=int(os.getenv("WORLD_SIZE")))

dtype = string_to_torch_dtype(args.dtype)
num_key_value_heads = 8

config = GPTBaseConfig(
    num_layers=2,
    position_embedding_type=args.position_embedding_type,
    vocab_size=50257,
    max_position_embeddings=512,
    hidden_size=128,
    normalization_function="layernorm",
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    rope_theta=10000,
    rope_scaling=None,
    rope_dim=None,
    m_emb=None,
    m_width=None,
    m_residual=None,
    init_method="normal",
    embedding_init_method="normal",
    use_depth_scaled_init=False,
    router_aux_loss_coef=0.001,
    tie_word_embeddings=False,
    sequence_mixer_blocks={
        "sequence_mixer_type": "softmax_attention",
        "add_bias": False,
        "num_attention_heads": 16,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": None,
        "softmax_dropout": 0,
        "dropout": 0,
        "attention_multiplier": None,
        "attention_multiplier_method": "1 / sqrt(head_dim)",
        "attention_gate": False,
        "exclusive_self_attention": False,
        "sliding_window": None,
    },
    mlp_blocks=[
        {"mlp_type": "MLP", "activation_function": "gelu_pytorch_tanh", "add_bias": False},
        {
            "mlp_type": "MoE",
            "activation_function": "gelu_pytorch_tanh",
            "add_bias": False,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "shared_intermediate_size": None,
        },
    ],
)

kernels = [Kernel.scattermoe]
if args.attention_implementation == "flash_attention_2":
    kernels.append(Kernel.flash_attention_2)
elif args.attention_implementation == "flash_attention_3":
    kernels.append(Kernel.flash_attention_3)
elif args.attention_implementation == "flash_attention_4":
    kernels.append(Kernel.flash_attention_4)

with enable_kernels(kernels):
    if torch.distributed.get_rank() == 0:
        with torch.device("meta"), ProcessGroupManager.set_dummy_tensor_parallel_world_size(1):
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
        # try sharding vocab matrices if really struggling for memory

        model_tp = AutoModelForCausalLM.from_config(
            config,
            use_padding_free_transformer=args.use_padding_free_transformer,
            sequence_parallel=args.sequence_parallel,
        )

    # copy to device without copying storage
    model_tp = model_tp.to_empty(device=torch.cuda.current_device())

    # load weights into tensor parallel model using SafeTensorsWeightsManager class
    # this avoids loading multiple copies of the parameters in CPU memory
    model_tp.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(args.tmp_path))

    # set model to eval mode
    model_tp = model_tp.to(dtype)
    model_tp.eval()

    Accelerator.set_seed(42)

    batch_size = 4
    sequence_length = 512

    input_ids = torch.randint(
        0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
    )
    labels = torch.randint(
        0, 50255, (batch_size, sequence_length), device=torch.cuda.current_device(), requires_grad=False
    )

    if args.use_padding_free_transformer:
        cu_seqlens = torch.arange(
            0, input_ids.numel() + 1, sequence_length, dtype=torch.int32, device=torch.cuda.current_device()
        )
        position_ids = torch.arange(0, sequence_length, 1, device=torch.cuda.current_device()).repeat(batch_size)

        output_tp = model_tp(
            input_ids=input_ids.view(-1),
            labels=labels.view(-1),
            cu_seqlens=cu_seqlens,
            max_seqlen=sequence_length,
            position_ids=position_ids,
        )
    else:
        output_tp = model_tp(input_ids=input_ids, labels=labels)

    loss_tp = output_tp.loss
    logits_tp = output_tp.logits[..., : config.vocab_size]

    if torch.distributed.get_rank() == 0:
        # loss computation hangs if we don't use dummy tensor parallel world size
        with ProcessGroupManager.set_dummy_tensor_parallel_world_size(1):
            output = model(input_ids=input_ids, labels=labels)

        loss = output.loss
        logits = output.logits

        if args.use_padding_free_transformer:
            logits_tp = logits_tp.reshape(batch_size, sequence_length, -1)

        error = (logits - logits_tp).abs().max()
        assert error < 5e-4, f"logits don't match for normal and tensor parallel model, error is ({error})"

        error = (loss - loss_tp).abs().max()
        assert error < 1e-3, f"losses don't match for normal and tensor parallel model, error is ({error})"
