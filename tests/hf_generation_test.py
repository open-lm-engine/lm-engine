# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
import subprocess
import sys
import tempfile

import pytest
import torch
from torch.testing import assert_close

from lm_engine.hf_generation import HFGPTBaseForCausalLM
from lm_engine.models import GPTBaseConfig, GPTBaseForCausalLM

from .utils import assert_equal_tensors, get_hybrid_m2rnn_test_config, skip_test_if_device_unavailable


def _get_model_and_inputs(
    device: torch.device, left_pad: int = 0
) -> tuple[HFGPTBaseForCausalLM, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    config = get_hybrid_m2rnn_test_config()
    model = HFGPTBaseForCausalLM(config).to(device)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(3, config.vocab_size, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)

    if left_pad > 0:
        input_ids[0, :left_pad] = config.pad_token_id
        attention_mask[0, :left_pad] = 0

    return model, input_ids, attention_mask


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_incremental_decode_matches_full_forward(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, _ = _get_model_and_inputs(device)

    with torch.no_grad():
        full_logits = model(input_ids=input_ids).logits

        output = model(input_ids=input_ids[:, :4], use_cache=True)
        step_logits = [output.logits]
        for t in range(4, input_ids.size(1)):
            output = model(input_ids=input_ids[:, t : t + 1], cache_params=output.cache_params, use_cache=True)
            step_logits.append(output.logits)

    assert_close(torch.cat(step_logits, dim=1), full_logits, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("left_pad", [0, 3])
def test_hf_greedy_generate_matches_native(device: torch.device, left_pad: int) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device, left_pad=left_pad)

    hf_output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, do_sample=False
    )
    native_output = GPTBaseForCausalLM.generate(
        model, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, temperature=0
    )
    assert_equal_tensors(hf_output, native_output, exact_match=True)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_hf_sampled_generate_matches_native(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device)

    torch.manual_seed(1234)
    hf_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.8,
        top_k=5,
    )

    torch.manual_seed(1234)
    native_output = GPTBaseForCausalLM.generate(
        model, input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16, temperature=0.8, top_k=5
    )

    assert_equal_tensors(hf_output, native_output, exact_match=True)


def test_from_pretrained_keeps_decay_gate_params_fp32() -> None:
    torch.manual_seed(42)
    config_dict = get_hybrid_m2rnn_test_config().to_dict()
    config_dict["tie_word_embeddings"] = True
    model = HFGPTBaseForCausalLM(GPTBaseConfig(**config_dict))

    with tempfile.TemporaryDirectory() as save_directory:
        model.save_pretrained(save_directory)
        loaded_model = HFGPTBaseForCausalLM.from_pretrained(save_directory, dtype=torch.bfloat16)

    assert loaded_model.transformer.wte.weight.dtype == torch.bfloat16
    for name, param in loaded_model.named_parameters():
        if "decay_gate" in name:
            assert param.dtype == torch.float32, name

    loaded_model.eval()
    with torch.no_grad():
        logits = loaded_model(input_ids=torch.randint(3, 128, (1, 8))).logits
    assert torch.isfinite(logits.float()).all()


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_logits_to_keep(device: torch.device) -> None:
    skip_test_if_device_unavailable(device)

    model, input_ids, attention_mask = _get_model_and_inputs(device)

    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        sliced_logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=3).logits

    assert_close(sliced_logits, full_logits[:, -3:])


def test_moe_training_forward_survives_external_torch_distributed() -> None:
    script = """
import torch
import torch.distributed

torch.distributed.init_process_group(backend="gloo")

from lm_engine.parallel import ProcessGroupManager

assert not ProcessGroupManager.is_initialized(), "meshes were never built"

from lm_engine.hf_generation import HFGPTBaseForCausalLM
from tests.utils import get_m2rnn_moe_test_config

model = HFGPTBaseForCausalLM(get_m2rnn_moe_test_config(num_layers=2))
model.train()
input_ids = torch.randint(3, model.config.vocab_size, (2, 16))
loss = model(input_ids=input_ids, labels=input_ids).loss
assert torch.isfinite(loss), f"non-finite loss {loss}"
"""

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env.update({"MASTER_ADDR": "localhost", "MASTER_PORT": "29513", "WORLD_SIZE": "1", "RANK": "0"})
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
