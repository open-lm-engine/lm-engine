# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

#!/usr/bin/env python3
"""
tracer_deltamlp.py – Activation and weight stats for a DeltaMLP model.

Logs to wandb: per-layer activation stats, per-chunk recurrent state evolution (S), and
weight stats. Forward hooks capture activations; successive prefix runs capture S at each
chunk boundary (causal property: S after token T is identical for any prefix ending at T).

Requires seq_len > 64 and chunk_size > 64
Usage:
    # freshly initialized model
    python scripts/tracer_deltamlp.py --config CONFIG --wandb_project PROJECT --wandb_run_name NAME
    # from a saved checkpoint (auto-detects latest step)
    python scripts/tracer_deltamlp.py --config CONFIG --checkpoint /data/checkpoints_jyo/... --wandb_project PROJECT --wandb_run_name NAME
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import yaml
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

import lm_engine.hf_models  # noqa: F401 – registers model classes with HF AutoModel
import wandb


def _progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def _load_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _load_pretrained_config(yaml_path: str) -> dict:
    data = _load_yaml(yaml_path)
    cfg = data["model_args"]["pretrained_config"]
    for key in ("layer_norm_epsilon", "initializer_range", "m_width", "m_emb", "m_residual"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = float(cfg[key])
    return cfg


def _get_real_input_ids(
    yaml_path: str,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    """Load a batch of real token IDs from the dataset specified in the YAML config.

    Uses the Megatron data pipeline directly (no distributed process group needed).
    The returned tensor is of shape [batch_size, seq_len] on CPU, dtype torch.long.
    If the dataset's sequence_length > seq_len, samples are truncated; if shorter,
    seq_len is capped to the dataset's sequence_length.
    """
    import numpy as np
    from transformers import AutoTokenizer

    from lm_engine.data.megatron.blended_megatron_dataset_builder import build
    from lm_engine.data.megatron.blended_megatron_dataset_config import GPTDatasetConfig

    data = _load_yaml(yaml_path)
    tokenizer_name = data["tokenizer_args"]["tokenizer_name"]
    class_args = data["datasets"][0]["class_args"]
    data_seq_len: int = class_args["sequence_length"]

    print(f"  Loading real data: {class_args.get('data_path', class_args.get('train_data_path'))}")
    print(f"  Dataset sequence_length={data_seq_len}, truncating to seq_len={seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # compile_helpers() calls Communication.barrier() which needs a process group.
    # Load the extension directly when running standalone.
    import lm_engine.data.megatron.utils as _mu

    if _mu._HELPERS is None:
        import os as _os

        from torch.utils.cpp_extension import load as _load_ext

        _helpers_dir = _os.path.dirname(_mu.__file__)
        _build_dir = _os.path.join(_helpers_dir, "build")
        _os.makedirs(_build_dir, exist_ok=True)
        _mu._HELPERS = _load_ext(
            "helpers",
            sources=_os.path.join(_helpers_dir, "helpers.cpp"),
            extra_cflags=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC", "-fdiagnostics-color"],
            build_directory=_build_dir,
            verbose=False,
        )

    # Request slightly more samples than needed so the dataset is valid; only train split is used.
    num_samples = max(batch_size * 4, 16)
    datasets = build(
        sizes=[num_samples, 1, 1],
        config=GPTDatasetConfig(
            sequence_length=data_seq_len,
            blend=class_args.get("data_path"),
            blend_per_split=[
                class_args.get("train_data_path"),
                class_args.get("val_data_path"),
                class_args.get("test_data_path"),
            ],
            split=class_args.get("split"),
            path_to_cache=class_args.get("data_cache_path"),
            fim_rate=class_args.get("fim_rate", 0),
            fim_spm_rate=class_args.get("fim_spm_rate", 0.5),
        ),
        tokenizer=tokenizer,
        node_uses_local_storage=class_args.get("node_uses_local_storage", False),
        random_seed=seed,
    )
    train_ds = datasets[0]  # [train, val, test]

    use_len = min(seq_len, data_seq_len)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=batch_size, replace=False)
    samples = [torch.from_numpy(train_ds[int(i)]["text"][:use_len].astype(np.int64)) for i in indices]
    return torch.stack(samples)  # [B, use_len]


def _shape_str(shape) -> str:
    return "x".join(str(d) for d in shape)


def _tensor_stats(
    t: torch.Tensor,
    log_dict: dict | None = None,
    prefix: str = "",
) -> None:
    if log_dict is None:
        return
    f = t.float()
    key = f"{prefix}[{_shape_str(t.shape)}]/"
    log_dict[f"{key}norm"] = f.norm().item()
    log_dict[f"{key}mean"] = f.mean().item()
    log_dict[f"{key}std"] = f.std().item()
    log_dict[f"{key}min"] = f.min().item()
    log_dict[f"{key}max"] = f.max().item()

    if f.dim() >= 2:
        per_token = f.norm(dim=-1)
        log_dict[f"{key}per_token_norm_mean"] = per_token.mean().item()
        log_dict[f"{key}per_token_norm_min"] = per_token.min().item()
        log_dict[f"{key}per_token_norm_max"] = per_token.max().item()


def _weight_stats(
    param: torch.nn.Parameter,
    log_dict: dict | None = None,
    prefix: str = "",
) -> None:
    if log_dict is None:
        return
    w = param.data.float()
    key = f"{prefix}[{_shape_str(param.shape)}]/"
    log_dict[f"{key}norm"] = w.norm().item()
    log_dict[f"{key}std"] = w.std().item()
    log_dict[f"{key}min"] = w.min().item()
    log_dict[f"{key}max"] = w.max().item()


def _slice_stats(t: torch.Tensor, start: int, end: int) -> dict[str, float]:
    """Per-token stats for t[:, start:end, ...], batch-averaged."""
    s = t[:, start:end].float().reshape(t.shape[0], (end - start), -1)  # [B, T', D]
    return {
        "norm": s.norm(dim=-1).mean().item(),  # mean per-token norm over batch
        "mean": s.mean().item(),
        "std": s.std().item(),
        "min": s.min().item(),
        "max": s.max().item(),
    }


def _output_hook(key: str, storage: dict):
    def hook(module, inp, output):
        t = output[0] if isinstance(output, tuple) else output
        if isinstance(t, torch.Tensor):
            storage[key] = t.detach().cpu()

    return hook


def _input_hook(key: str, storage: dict):
    def hook(module, inp, output):
        t = inp[0]
        if isinstance(t, torch.Tensor):
            storage[key] = t.detach().cpu()

    return hook


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, step: int | None = None) -> int:
    """Load model weights from a DCP checkpoint directory into a bare HF model.

    Args:
        model: freshly built AutoModelForCausalLM (no ModelWrapper).
        checkpoint_path: base save path (contains latest_checkpointed_iteration.json)
                         OR a direct step path (…/global_stepN).
        step: if None, auto-detected from latest_checkpointed_iteration.json.

    Returns:
        The iteration (step) that was loaded.
    """
    # Resolve to the step-level directory.
    latest_json = os.path.join(checkpoint_path, "latest_checkpointed_iteration.json")
    if os.path.exists(latest_json):
        # checkpoint_path is the base path; find the step.
        if step is None:
            step = json.load(open(latest_json))["latest_checkpointed_iteration"]
        step_dir = os.path.join(checkpoint_path, f"global_step{step}")
    else:
        # checkpoint_path already points at a step directory (e.g. …/global_step8000).
        step_dir = checkpoint_path
        step = int(os.path.basename(step_dir).replace("global_step", "")) if step is None else step

    model_dir = os.path.join(step_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No model directory found at {model_dir}")

    print(f"  Loading checkpoint from {model_dir} (step {step})")

    # Load the raw state dict from the distcp shards (no distributed process group needed).
    state: dict = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(model_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    raw = state["state"]  # keys are "model.<param_name>"

    # Strip the "model." prefix added by ModelWrapper so keys match the HF model.
    stripped = {k.removeprefix("model."): v for k, v in raw.items()}

    # strict=True raises RuntimeError on missing/unexpected keys.
    model.load_state_dict(stripped, strict=True)

    print(f"  Checkpoint loaded successfully ({len(stripped)} tensors, step {step})")
    return step


def run_diagnostic(
    yaml_path: str,
    wandb_project: str,
    wandb_run_name: str,
    seq_len: int = 128,
    batch_size: int = 1,
    seed: int = 42,
    chunk_size: int = 128,
    checkpoint_path: str | None = None,
    checkpoint_step: int | None = None,
    use_real_data: bool = False,
) -> None:
    if seq_len <= 64:
        raise ValueError(
            f"seq_len={seq_len} must be > 64. DeltaMLP selects fused_recurrent mode for "
            "q_len <= 64 in eval, which is not yet implemented. Use seq_len >= 65."
        )
    if chunk_size <= 64:
        raise ValueError(
            f"chunk_size={chunk_size} must be > 64. Each prefix run uses q_len=chunk_size "
            "and DeltaMLP selects fused_recurrent mode for q_len <= 64. Use chunk_size >= 65."
        )

    torch.manual_seed(seed)

    # ---- Build model --------------------------------------------------------
    pretrained_config = _load_pretrained_config(yaml_path)
    config = AutoConfig.for_model(**pretrained_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_config(config)

    loaded_step: int | None = None
    if checkpoint_path is not None:
        # Load on CPU first, then move — avoids needing the right device before weights are set.
        loaded_step = _load_checkpoint(model, checkpoint_path, step=checkpoint_step)

    model.eval()
    model.to(device=device, dtype=dtype)

    vocab_size: int = config.vocab_size
    hidden_size: int = config.hidden_size
    num_layers: int = config.num_layers
    tied: bool = config.tie_word_embeddings

    first_delta = model.transformer.h["0"].mlp_block
    mlb0 = pretrained_config["mlp_blocks"][0]

    # asserts to check if the configs are compatible with the tracer
    assert first_delta.use_input_gate
    assert not first_delta.use_mlp_stream

    print("=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    print(
        f"  weights                  : {'checkpoint step ' + str(loaded_step) if loaded_step is not None else 'random init'}"
    )
    print(f"  input data               : {'real (from config)' if use_real_data else 'random token IDs'}")
    print(f"  model_type               : {pretrained_config['model_type']}")
    print(f"  hidden_size              : {hidden_size}")
    print(f"  num_layers               : {num_layers}")
    print(f"  vocab_size               : {vocab_size}")
    print(f"  init_method              : {config.init_method}")
    print(f"  initializer_range        : {pretrained_config.get('initializer_range', 'N/A')}")
    print(f"  m_width                  : {getattr(config, 'm_width', 'N/A')}")
    print(f"  m_emb                    : {getattr(config, 'm_emb', 'N/A')}")
    print(f"  tied_embeddings          : {tied}")
    print(f"  input shape              : ({batch_size}, {seq_len})")
    print(f"  use_depth_scaled_init    : {getattr(config, 'use_depth_scaled_init', False)}")
    print(f"  use_depth_scaled_init_δ  : {getattr(config, 'use_depth_scaled_init_delta', False)}")
    print(f"  -- DeltaMLP (layer 0) --")
    print(f"  use_v_proj               : {first_delta.use_v_proj}")
    print(f"  use_q_l2norm             : {first_delta.use_q_l2norm}")
    print(f"  use_shortconv            : {first_delta.use_shortconv}")
    print(f"  use_mlp_stream           : {first_delta.use_mlp_stream}")
    print(f"  use_input_gate           : {first_delta.use_input_gate}")
    print(f"  use_output_gate          : {first_delta.use_output_gate}")
    print(f"  use_output_norm          : {first_delta.use_output_norm}")
    print(f"  use_zero_init_k          : {mlb0.get('use_zero_init_k', False)}")
    print(f"  beta_per_head            : {mlb0.get('beta_per_head', False)}")
    print(f"  conv_size                : {first_delta.conv_size}")

    # ---- wandb init ---------------------------------------------------------
    log_dict: dict = {}
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model_type": pretrained_config["model_type"],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "init_method": config.init_method,
            "initializer_range": pretrained_config.get("initializer_range"),
            "m_width": getattr(config, "m_width", None),
            "m_emb": getattr(config, "m_emb", None),
            "tied_embeddings": tied,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "seed": seed,
            "chunk_size": chunk_size,
            "use_depth_scaled_init": getattr(config, "use_depth_scaled_init", False),
            "use_depth_scaled_init_delta": getattr(config, "use_depth_scaled_init_delta", False),
            "use_v_proj": first_delta.use_v_proj,
            "use_q_l2norm": first_delta.use_q_l2norm,
            "use_shortconv": first_delta.use_shortconv,
            "use_mlp_stream": first_delta.use_mlp_stream,
            "use_input_gate": first_delta.use_input_gate,
            "use_output_gate": first_delta.use_output_gate,
            "use_output_norm": first_delta.use_output_norm,
            "use_zero_init_k": mlb0.get("use_zero_init_k", False),
            "beta_per_head": mlb0.get("beta_per_head", False),
            "conv_size": first_delta.conv_size,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": loaded_step,
            "use_real_data": use_real_data,
        },
    )

    # ---- Token IDs ----------------------------------------------------------
    if use_real_data:
        input_ids = _get_real_input_ids(yaml_path, batch_size, seq_len, seed).to(device)
        # Actual length may be capped to the dataset's sequence_length.
        seq_len = input_ids.shape[1]
        print(f"  Real data loaded: input_ids shape = {tuple(input_ids.shape)}")
    else:
        rng = torch.Generator()
        rng.manual_seed(seed)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng).to(device)

    # ---- Register hooks -----------------------------------------------------
    captures: dict[str, torch.Tensor] = {}
    handles = []

    handles.append(model.transformer.wte.register_forward_hook(_output_hook("embedding", captures)))

    for i in range(num_layers):
        block = model.transformer.h[str(i)]
        delta = block.mlp_block

        # block input / residual stream entering this layer
        handles.append(block.ln_1.register_forward_hook(_input_hook(f"L{i}.residual_in", captures)))
        handles.append(block.ln_1.register_forward_hook(_output_hook(f"L{i}.ln1_out", captures)))

        # sequence mixer output (attention, before residual add)
        handles.append(block.sequence_mixer.register_forward_hook(_output_hook(f"L{i}.attn_out", captures)))

        # residual after attention → DeltaMLP input
        handles.append(block.ln_2.register_forward_hook(_input_hook(f"L{i}.residual_post_attn", captures)))
        handles.append(block.ln_2.register_forward_hook(_output_hook(f"L{i}.ln2_out", captures)))

        # ---- DeltaMLP internals ----
        # q_proj: [B, T, key_dim] or [B, T, key_dim*2] when use_input_gate
        handles.append(delta.q_proj.register_forward_hook(_output_hook(f"L{i}.delta.q_proj", captures)))
        # k_proj (LowRankLinear): [B, T, key_dim]
        handles.append(delta.k_proj.register_forward_hook(_output_hook(f"L{i}.delta.k_proj", captures)))
        # v_proj (LowRankLinear, if present): [B, T, value_dim]
        if delta.use_v_proj:
            handles.append(delta.v_proj.register_forward_hook(_output_hook(f"L{i}.delta.v_proj", captures)))
        # bg_proj: [B, T, num_v_heads] (+ value_dim if use_output_gate)
        handles.append(delta.bg_proj.register_forward_hook(_output_hook(f"L{i}.delta.bg_proj", captures)))
        # beta: sigmoid(bg) after head-repeat, [B, T, num_heads] – use side-channel (inline op, no module)
        delta._capture_beta = True
        # kv post-act/conv: captures k and v after activation (and conv when use_shortconv).
        # Enabled unconditionally so update_frob always uses post-activation values.
        delta._capture_kv_post_conv = True
        # o_norm: capture both input (pre-norm) and output (post-norm), shape [B, T, num_heads * v_head_dim]
        if delta.use_output_norm:
            handles.append(delta.o_norm.register_forward_hook(_input_hook(f"L{i}.delta.o_pre_norm", captures)))
            handles.append(delta.o_norm.register_forward_hook(_output_hook(f"L{i}.delta.o_post_norm", captures)))
        # DeltaMLP output: [B, T, hidden_size] before residual add
        handles.append(delta.register_forward_hook(_output_hook(f"L{i}.delta.out", captures)))

        # block output = residual stream after DeltaMLP residual add
        handles.append(block.register_forward_hook(_output_hook(f"L{i}.residual_post_delta", captures)))

    handles.append(model.transformer.ln_f.register_forward_hook(_output_hook("ln_f_out", captures)))

    # ---- Forward pass -------------------------------------------------------
    # model.eval() is set; seq_len > 64 ensures DeltaMLP uses chunk mode.
    # use_cache=False prevents HF from creating a cache with zeroed conv_state,
    # which would route causal_convolution into the one-token inference path.
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()

    # Collect side-channel captures and clean up flags + cached tensors.
    for i in range(num_layers):
        delta = model.transformer.h[str(i)].mlp_block
        captures[f"L{i}.delta.beta"] = delta._last_beta.cpu()
        del delta._capture_beta
        del delta._last_beta
        # kv_post_act: always captured (post-activation, post-conv when use_shortconv).
        captures[f"L{i}.delta.kv_post_act"] = delta._last_kv_post_conv.cpu()
        del delta._capture_kv_post_conv
        del delta._last_kv_post_conv

    lm_logits: torch.Tensor = output.logits.detach().cpu()  # [B, T, V]

    # =========================================================================
    # ACTIVATION STATS
    # =========================================================================

    ld = log_dict

    _tensor_stats(captures["embedding"], log_dict=ld, prefix="acts/embedding")

    for i in _progress(range(num_layers), desc="Activation stats", unit="layer"):
        delta = model.transformer.h[str(i)].mlp_block
        p = f"acts/layer_{i}"

        _tensor_stats(captures[f"L{i}.residual_in"], log_dict=ld, prefix=f"{p}/residual_in")
        _tensor_stats(captures[f"L{i}.ln1_out"], log_dict=ld, prefix=f"{p}/ln1_out")
        _tensor_stats(captures[f"L{i}.attn_out"], log_dict=ld, prefix=f"{p}/attn_out")
        _tensor_stats(captures[f"L{i}.residual_post_attn"], log_dict=ld, prefix=f"{p}/residual_post_attn")
        _tensor_stats(captures[f"L{i}.ln2_out"], log_dict=ld, prefix=f"{p}/ln2_out")
        _tensor_stats(captures[f"L{i}.delta.q_proj"], log_dict=ld, prefix=f"{p}/delta/q_proj")
        _tensor_stats(captures[f"L{i}.delta.k_proj"], log_dict=ld, prefix=f"{p}/delta/k_proj")
        if delta.use_v_proj:
            _tensor_stats(captures[f"L{i}.delta.v_proj"], log_dict=ld, prefix=f"{p}/delta/v_proj")
        _tensor_stats(captures[f"L{i}.delta.bg_proj"], log_dict=ld, prefix=f"{p}/delta/bg_proj")
        _tensor_stats(captures[f"L{i}.delta.beta"], log_dict=ld, prefix=f"{p}/delta/beta")
        # Per-head beta stats (over full sequence).
        # When beta_per_head=False all heads share the same value (repeated), so variance=0.
        beta_full = captures[f"L{i}.delta.beta"].float()  # [B, T, H]
        for h in range(beta_full.shape[-1]):
            bh = beta_full[:, :, h]  # [B, T]
            ld[f"{p}/delta/beta_head_{h}/mean"] = bh.mean().item()
            ld[f"{p}/delta/beta_head_{h}/std"] = bh.std().item()
            ld[f"{p}/delta/beta_head_{h}/min"] = bh.min().item()
            ld[f"{p}/delta/beta_head_{h}/max"] = bh.max().item()
        # Average variance of beta across heads per time step: E_t[ Var_h(beta_t) ]
        ld[f"{p}/delta/beta_head_var_mean"] = beta_full.var(dim=-1).mean().item()
        if delta.use_shortconv:
            _tensor_stats(captures[f"L{i}.delta.kv_post_act"], log_dict=ld, prefix=f"{p}/delta/kv_post_act")
        if delta.use_output_norm:
            _tensor_stats(captures[f"L{i}.delta.o_pre_norm"], log_dict=ld, prefix=f"{p}/delta/o_pre_norm")
            _tensor_stats(captures[f"L{i}.delta.o_post_norm"], log_dict=ld, prefix=f"{p}/delta/o_post_norm")
        _tensor_stats(captures[f"L{i}.delta.out"], log_dict=ld, prefix=f"{p}/delta/out")
        _tensor_stats(captures[f"L{i}.residual_post_delta"], log_dict=ld, prefix=f"{p}/residual_post_delta")

    _tensor_stats(captures["ln_f_out"], log_dict=ld, prefix="acts/ln_f_out")

    lm_probs = F.softmax(lm_logits.float(), dim=-1)
    _tensor_stats(lm_logits, log_dict=ld, prefix="acts/lm_logits")
    _tensor_stats(lm_probs, log_dict=ld, prefix="acts/lm_probs")

    # =========================================================================
    # PER-CHUNK HIDDEN STATE EVOLUTION
    # =========================================================================
    #
    # Run the model on successive input prefixes (causal property guarantees
    # that the recurrent state after token T is the same whether we run on the
    # full sequence or just the prefix up to T). Each prefix run stores the
    # final recurrent state via the _capture_recurrent_state side-channel.
    # =========================================================================

    num_chunks = math.ceil(seq_len / chunk_size)

    # Enable recurrent state capture on all DeltaMLP layers.
    for i in range(num_layers):
        model.transformer.h[str(i)].mlp_block._capture_recurrent_state = True

    # Collect per-layer lists across chunks.
    ssm_norms: list[list[float]] = [[] for _ in range(num_layers)]
    ssm_means: list[list[float]] = [[] for _ in range(num_layers)]
    ssm_stds: list[list[float]] = [[] for _ in range(num_layers)]
    ssm_mins: list[list[float]] = [[] for _ in range(num_layers)]
    ssm_maxs: list[list[float]] = [[] for _ in range(num_layers)]
    ssm_sizes: list[int] = [0] * num_layers  # flat H*K*V size per layer, set on first chunk

    for c in _progress(range(num_chunks), desc="Chunk state", unit="chunk"):
        end = min((c + 1) * chunk_size, seq_len)
        with torch.no_grad():
            model(input_ids=input_ids[:, :end], use_cache=False)
        for i in range(num_layers):
            delta = model.transformer.h[str(i)].mlp_block
            S = delta._last_recurrent_state.float()  # [B, H, K, V]
            s_flat = S.reshape(batch_size, -1)
            if c == 0:
                ssm_sizes[i] = s_flat.shape[-1]
            ssm_norms[i].append(s_flat.norm(dim=-1).mean().item())
            ssm_means[i].append(s_flat.mean(dim=-1).mean().item())
            ssm_stds[i].append(s_flat.std(dim=-1).mean().item())
            ssm_mins[i].append(s_flat.min().item())
            ssm_maxs[i].append(s_flat.max().item())

    # Disable capture and free cached tensors.
    for i in range(num_layers):
        delta = model.transformer.h[str(i)].mlp_block
        del delta._capture_recurrent_state
        del delta._last_recurrent_state

    # =========================================================================
    # WEIGHT STATISTICS
    # =========================================================================
    # Computed here so they are included in the step=0 wandb.log below.

    _weight_stats(model.transformer.wte.weight, log_dict=ld, prefix="weights/embedding/wte")

    for i in _progress(range(num_layers), desc="Weight stats ", unit="layer"):
        block = model.transformer.h[str(i)]
        delta = block.mlp_block
        attn = block.sequence_mixer
        wp = f"weights/layer_{i}"

        # sequence mixer weights (generic – log whatever linear layers exist)
        for attr_name in ("c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            submod = getattr(attn, attr_name, None)
            if submod is not None and hasattr(submod, "weight"):
                _weight_stats(submod.weight, log_dict=ld, prefix=f"{wp}/attn/{attr_name}")

        # DeltaMLP weights
        _weight_stats(delta.q_proj.weight, log_dict=ld, prefix=f"{wp}/delta/q_proj")
        _weight_stats(delta.k_proj.u_proj.weight, log_dict=ld, prefix=f"{wp}/delta/k_proj_u")
        _weight_stats(delta.k_proj.v_proj.weight, log_dict=ld, prefix=f"{wp}/delta/k_proj_v")
        if delta.use_v_proj:
            _weight_stats(delta.v_proj.u_proj.weight, log_dict=ld, prefix=f"{wp}/delta/v_proj_u")
            _weight_stats(delta.v_proj.v_proj.weight, log_dict=ld, prefix=f"{wp}/delta/v_proj_v")
        _weight_stats(delta.bg_proj.weight, log_dict=ld, prefix=f"{wp}/delta/bg_proj")
        # initial_state.weight IS S_0: the initial recurrent hidden state matrix.
        # It is used via repeat() directly, never called as a linear forward.
        _weight_stats(delta.initial_state.weight, log_dict=ld, prefix=f"{wp}/delta/initial_state")
        if delta.use_shortconv:
            _weight_stats(delta.kv_conv1d.weight, log_dict=ld, prefix=f"{wp}/delta/kv_conv1d")

    if not tied:
        _weight_stats(model.lm_head.weight, log_dict=ld, prefix="weights/lm_head")

    # ---- Per-chunk wandb charts -------------------------------------------
    # Each quantity is logged with wandb step = chunk index so wandb auto-plots
    # line charts with chunk on the x-axis. S is logged starting at step 0
    # (= S_0, the learned initial state) then at step c+1 after chunk c.
    # All activation quantities are logged at the step of the chunk they came from.
    # step 0: S_0 stats from initial_state.weight (same values as the
    # repeated state, since repeat doesn't change per-element statistics).
    s0_log: dict[str, float] = {}
    for i in range(num_layers):
        delta = model.transformer.h[str(i)].mlp_block
        w = delta.initial_state.weight.float().reshape(-1)
        for stat, val in zip(
            ("norm", "mean", "std", "min", "max"),
            (w.norm().item(), w.mean().item(), w.std().item(), w.min().item(), w.max().item()),
        ):
            s0_log[f"chunk/layer_{i}/S[{ssm_sizes[i]}]/{stat}"] = val
    wandb.log({**s0_log, **log_dict}, step=0)

    # log at token count boundaries: step = number of tokens processed so far
    for c in range(num_chunks):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, seq_len)
        step = end  # x-axis = tokens processed
        chunk_log: dict[str, float] = {}

        for i in range(num_layers):
            delta = model.transformer.h[str(i)].mlp_block
            p = f"chunk/layer_{i}"

            # recurrent state after this chunk
            for stat, vals in zip(
                ("norm", "mean", "std", "min", "max"),
                (ssm_norms[i], ssm_means[i], ssm_stds[i], ssm_mins[i], ssm_maxs[i]),
            ):
                chunk_log[f"{p}/S[{ssm_sizes[i]}]/{stat}"] = vals[c]

            # activation slices – all from the single main forward pass
            # key format: {p}/{name}[D]/{stat}  where D is the feature dim
            def _log_slice(name: str, t: torch.Tensor) -> None:
                d = t.shape[-1]
                for stat, val in _slice_stats(t, start, end).items():
                    chunk_log[f"{p}/{name}[{d}]/{stat}"] = val

            _log_slice("hidden_state", captures[f"L{i}.residual_in"])
            _log_slice("q", captures[f"L{i}.delta.q_proj"])
            _log_slice("k_pre_conv", captures[f"L{i}.delta.k_proj"])

            # v_pre_conv: ln2_out when use_v_proj=False because v = hidden_states directly
            v_pre_key = f"L{i}.delta.v_proj" if delta.use_v_proj else f"L{i}.ln2_out"
            _log_slice("v_pre_conv", captures[v_pre_key])

            if delta.use_shortconv:
                kv = captures[f"L{i}.delta.kv_post_act"]
                _log_slice("k_post_conv", kv[:, :, : delta.key_dim])
                _log_slice("v_post_conv", kv[:, :, delta.key_dim :])

            _log_slice("beta", captures[f"L{i}.delta.beta"])

            # Per-head beta: mean per head within this chunk's time window.
            # When beta_per_head=False all heads share the same repeated value (var=0).
            beta_s = captures[f"L{i}.delta.beta"][:, start:end].float()  # [B, T', H]
            for h in range(beta_s.shape[-1]):
                chunk_log[f"{p}/beta_head_{h}/mean"] = beta_s[:, :, h].mean().item()
            # E_t[ Var_h(beta_t) ]: average over time of variance across heads per step.
            chunk_log[f"{p}/beta_head_var_mean"] = beta_s.var(dim=-1).mean().item()

            if delta.use_output_norm:
                _log_slice("o_pre_norm", captures[f"L{i}.delta.o_pre_norm"])
                _log_slice("o_post_norm", captures[f"L{i}.delta.o_post_norm"])

            _log_slice("o_out", captures[f"L{i}.delta.out"])

            # ---- update Frobenius norm: ||beta_t * v_t * k_t^T||_F per token ----
            # = beta_t * ||v_t||_2 * ||k_t||_2 per head, then Frobenius over all heads
            # directly comparable to S[N]/norm
            # kv_post_act is always captured (post-activation, post-conv when use_shortconv).
            H = beta_s.shape[-1]
            kv_s = captures[f"L{i}.delta.kv_post_act"][:, start:end].float()
            k_s = kv_s[:, :, : delta.key_dim]
            v_s = kv_s[:, :, delta.key_dim :]
            # reshape k to per-head: [B, T', H, K], compute per-head norms
            k_per_head = k_s.reshape(k_s.shape[0], k_s.shape[1], H, -1)  # [B, T', H, K]
            # normalize k per head (matches use_k_l2norm_in_kernel=True in chunk_delta_rule)
            k_per_head_norm = k_per_head / (k_per_head.norm(dim=-1, keepdim=True) + 1e-6)
            k_norms = k_per_head_norm.norm(dim=-1)  # [B, T', H] ≈ 1.0
            # v is broadcast over all k-heads (num_v_heads=1): norm is [B, T', 1]
            v_norms = v_s.norm(dim=-1, keepdim=True)  # [B, T', 1]
            # per-head update norm, then Frobenius over all heads → [B, T']
            update_per_head = beta_s * v_norms * k_norms  # [B, T', H]
            update_frob = update_per_head.pow(2).sum(dim=-1).sqrt()  # [B, T']
            update_size = k_s.shape[-1] * v_s.shape[-1]  # H*K × V = full update matrix elements
            chunk_log[f"{p}/update_frob[{update_size}]/mean"] = update_frob.mean().item()
            chunk_log[f"{p}/update_frob[{update_size}]/std"] = update_frob.std().item()
            chunk_log[f"{p}/update_frob[{update_size}]/min"] = update_frob.min().item()
            chunk_log[f"{p}/update_frob[{update_size}]/max"] = update_frob.max().item()

        wandb.log(chunk_log, step=step)

    # ---- wandb finish ----------------------------------------------------------
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation and weight stats of a freshly initialized DeltaMLP model."
    )
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length. Must be > 64 (DeltaMLP fused_recurrent not implemented).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Chunk size for per-chunk state reporting (default: 128). Must be > 64.",
    )
    parser.add_argument("--wandb_project", type=str, required=True, help="wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="wandb run name.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint base directory (contains latest_checkpointed_iteration.json) "
            "or directly to a step directory (…/global_stepN). "
            "If omitted, uses randomly initialized weights."
        ),
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Step to load from --checkpoint base path. Defaults to latest if not set.",
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        default=False,
        help=(
            "Load real token IDs from the dataset specified in the config's 'datasets' section "
            "instead of using random token IDs. Requires the data files to be accessible."
        ),
    )
    args = parser.parse_args()
    run_diagnostic(
        yaml_path=args.config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_path=args.checkpoint,
        checkpoint_step=args.checkpoint_step,
        use_real_data=args.use_real_data,
    )


if __name__ == "__main__":
    main()
