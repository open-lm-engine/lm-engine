# **************************************************
# Copyright (c) 2026, Mayank Mishra, Zhonglin Han
# **************************************************

"""E2E training smoke tests.

Pytest launches torchrun, and torchrun re-enters this module as __main__ for
each worker rank. The worker uses the production pretrain entrypoint with a
smoke-sized config and synthetic data.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Thread
from unittest.mock import patch
from uuid import uuid4

import pytest
import torch

import lm_engine.hf_models  # noqa: F401
from lm_engine.arguments import TrainingArgs
from lm_engine.logging_utils import log_rank_0
from lm_engine.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER_SWA_MUONH_CONFIG = str(REPO_ROOT / "configs/research/delta-mlp/transformer-swa-muonH.yml")
NUM_DEVICES = 8
NUM_LAYERS = 12
NUM_STEPS = 1
TORCHRUN_TIMEOUT_SECONDS = int(os.environ.get("LM_ENGINE_E2E_TIMEOUT_SECONDS", "3600"))
DUMP_ROOT = "/tmp/torch_compile_dump"
_DELETE = object()
INDUCTOR_DUMP_ENV = {
    "TORCH_COMPILE_DEBUG": "1",
}


@pytest.mark.gpu
@pytest.mark.slow
def test_muonh_e2e_8gpu(tmp_path: Path) -> None:
    if torch.cuda.device_count() < NUM_DEVICES:
        pytest.skip(f"smoke test requires {NUM_DEVICES} CUDA devices")

    env = os.environ.copy()
    test_name = "test_muonh_e2e_8gpu"
    dump_dir = _dump_dir(env, test_name) if _env_flag(env, "LM_ENGINE_INDUCTOR_DUMP") else None
    if dump_dir is not None:
        _set_inductor_dump_env(env, dump_dir)
        print(f"Inductor dump dir: {dump_dir}", flush=True)

    sentinel_file = tmp_path / "passed"
    output = _run_torchrun(
        _torchrun_cmd(
            env,
            config_path=TRANSFORMER_SWA_MUONH_CONFIG,
            num_layers=NUM_LAYERS,
            num_steps=NUM_STEPS,
            dump_dir=dump_dir,
            sentinel_file=sentinel_file,
        ),
        env,
    )
    assert sentinel_file.exists(), f"worker did not write success sentinel\n{output}"


def _env_flag(env: dict[str, str], name: str) -> bool:
    return env.get(name, "").lower() in {"1", "true", "yes", "on"}


def _default_dump_dir(test_name: str) -> str:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    return os.path.join(DUMP_ROOT, run_id, test_name)


def _dump_dir(env: dict[str, str], test_name: str) -> str:
    return env.get("LM_ENGINE_INDUCTOR_DUMP_DIR", _default_dump_dir(test_name))


def _torchrun_cmd(
    env: dict[str, str],
    *,
    config_path: str,
    num_layers: int,
    num_steps: int,
    dump_dir: str | None,
    sentinel_file: Path,
) -> list[str]:
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_DEVICES}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "-m",
        "tests.training.integration_test",
        "--config",
        config_path,
        "--num-layers",
        str(num_layers),
        "--num-steps",
        str(num_steps),
        "--sentinel-file",
        str(sentinel_file),
    ]

    if _env_flag(env, "LM_ENGINE_INDUCTOR_DUMP"):
        cmd.append("--dump")
        assert dump_dir is not None
        cmd += ["--dump-dir", dump_dir]

    return cmd


def _run_torchrun(cmd: list[str], env: dict[str, str]) -> str:
    output_lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None

    def _stream_output() -> None:
        for line in process.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)

    output_thread = Thread(target=_stream_output, daemon=True)
    output_thread.start()

    try:
        returncode = process.wait(timeout=TORCHRUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output_thread.join(timeout=5)
        pytest.fail(f"torchrun timed out after {TORCHRUN_TIMEOUT_SECONDS}s\n{''.join(output_lines)}")

    output_thread.join(timeout=5)
    output = "".join(output_lines)
    if returncode != 0:
        pytest.fail(f"torchrun failed with exit code {returncode}\n{output}")

    return output


def _enable_inductor_dump(dump: bool, dump_dir: str) -> None:
    if not dump:
        return
    os.makedirs(dump_dir, exist_ok=True)
    _set_inductor_dump_env(os.environ, dump_dir)


def _set_inductor_dump_env(env: dict[str, str], dump_dir: str) -> None:
    dump_env = {
        **INDUCTOR_DUMP_ENV,
        "TORCH_COMPILE_DEBUG_DIR": dump_dir,
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(dump_dir, "inductor_cache"),
    }
    if _env_flag(env, "LM_ENGINE_INDUCTOR_VERBOSE"):
        dump_env["TORCH_LOGS"] = "+inductor,+output_code"
    env.update(dump_env)


def _merge_config(config: dict, override: dict) -> dict:
    merged = deepcopy(config)
    for key, value in override.items():
        if value is _DELETE:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def _config_override(config: dict, *, num_layers: int, sequence_length: int | None, num_steps: int) -> dict:
    # The production pretraining dataloader only supports one dataset, so keep
    # the smoke override honest and avoid silently dropping extra datasets.
    assert len(config["datasets"]) == 1

    pretrained_config = config["model_args"]["pretrained_config"]

    def _truncated_blocks(name: str, block_override: dict | None = None) -> list[dict]:
        blocks = pretrained_config[name]
        assert len(blocks) >= num_layers, f"{name} has {len(blocks)} blocks, need {num_layers}"
        if block_override is None:
            return blocks[:num_layers]
        return [_merge_config(block, block_override) for block in blocks[:num_layers]]

    override = {
        "kernel_args": {
            "kernels": [],
        },
        "model_args": {
            "pretrained_config": {
                "num_layers": num_layers,
                "sequence_mixer_blocks": _truncated_blocks("sequence_mixer_blocks", {"sliding_window": _DELETE}),
                "mlp_blocks": _truncated_blocks("mlp_blocks"),
            },
        },
        "training_parameters": {
            "num_training_steps": num_steps,
            "gradient_accumulation_steps": 1,
            "eval_during_training": False,
        },
        "logging_args": {
            "log_interval": 1,
            "experiments_tracker_name": None,
        },
        "load_args": _DELETE,
    }

    if sequence_length is not None:
        override["datasets"] = [
            _merge_config(
                config["datasets"][0],
                {"class_args": {"sequence_length": sequence_length}},
            )
        ]
        override["model_args"]["pretrained_config"]["max_position_embeddings"] = max(
            pretrained_config["max_position_embeddings"],
            sequence_length,
        )

    gradient_checkpointing_args = config.get("distributed_args", {}).get("gradient_checkpointing_args", {})
    if "num_blocks" in gradient_checkpointing_args:
        override["distributed_args"] = {
            "gradient_checkpointing_args": {
                "num_blocks": min(gradient_checkpointing_args["num_blocks"], num_layers),
            },
        }

    return override


def _synthetic_train_dataloader(args: TrainingArgs) -> Iterator[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(args.random_args.seed)
    shape = (
        args.training_parameters.micro_batch_size,
        args.datasets[0].class_args["sequence_length"] + 1,
    )
    vocab_size = args.model_args.pretrained_config["vocab_size"]

    while True:
        yield {
            "text": torch.randint(
                low=0,
                high=vocab_size,
                size=shape,
                generator=generator,
                dtype=torch.long,
            )
        }


def _synthetic_pretraining_dataloaders(
    args: TrainingArgs, *_
) -> tuple[Iterator[dict[str, torch.Tensor]], list[None], list[None]]:
    return _synthetic_train_dataloader(args), [], []


def _load_smoke_args(config_path: str, num_layers: int, sequence_length: int | None, num_steps: int) -> TrainingArgs:
    config = load_yaml(config_path)
    override = _config_override(
        config,
        num_layers=num_layers,
        sequence_length=sequence_length,
        num_steps=num_steps,
    )
    return TrainingArgs(**_merge_config(config, override))


def _parse_worker_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=TRANSFORMER_SWA_MUONH_CONFIG)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="override datasets[0].class_args.sequence_length (default: keep YAML value)",
    )
    p.add_argument("--num-steps", type=int, default=NUM_STEPS)
    p.add_argument(
        "--dump",
        action="store_true",
        help="enable inductor codegen dump; pre-set env vars are NOT overridden",
    )
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--sentinel-file", default=None)
    return p.parse_args()


def _run_pretrain(args: TrainingArgs) -> None:
    import lm_engine.pretrain as pretrain

    with (
        patch.object(pretrain, "get_args", return_value=args),
        patch.object(pretrain, "get_pretraining_dataloaders", _synthetic_pretraining_dataloaders),
        # This E2E smoke covers the training path. Checkpoint save/reload should
        # stay in a separate test so this remains fast and side-effect-light.
        patch.object(pretrain, "save_checkpoint"),
    ):
        # pretrain.main expects an args class and calls get_args(args_class).
        # The patched get_args returns the smoke TrainingArgs above.
        pretrain.main(TrainingArgs)


def _write_sentinel(path: str | None) -> None:
    if path is not None and os.environ.get("RANK", "0") == "0":
        Path(path).write_text("Passed\n")


def _e2e_worker() -> None:
    cli = _parse_worker_args()
    _enable_inductor_dump(cli.dump, cli.dump_dir or _default_dump_dir("manual"))

    args = _load_smoke_args(cli.config, cli.num_layers, cli.sequence_length, cli.num_steps)
    _run_pretrain(args)

    _write_sentinel(cli.sentinel_file)
    log_rank_0(logging.INFO, "Passed")


if __name__ == "__main__":
    _e2e_worker()
    sys.exit(0)
