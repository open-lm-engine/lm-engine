# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
import tempfile

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from lm_engine.hf_models import export_to_huggingface

from ..test_common import TestCommons


class GenerationTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            ["mha", "mqa"],
            ["learned_absolute"],
            [torch.float32],
            [True, False],
        )
    )
    def test_generation_matches_bigcode(
        self,
        device: torch.device,
        attention_head_type: str,
        position_embedding_type: str,
        torch_dtype: torch.dtype,
        use_cache: bool,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        lm_engine_config = self.get_dense_test_config(attention_head_type, position_embedding_type)
        lm_engine_config.use_cache = use_cache

        lm_engine_model = self.from_config(lm_engine_config, torch_dtype=torch_dtype).to(device)
        lm_engine_model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            lm_engine_path = os.path.join(tmpdir, "lm_engine")
            bigcode_path = os.path.join(tmpdir, "bigcode")

            lm_engine_model.save_pretrained(lm_engine_path)
            export_to_huggingface(lm_engine_path, bigcode_path, model_type="gpt_bigcode")

            bigcode_model = AutoModelForCausalLM.from_pretrained(bigcode_path).to(device)
            bigcode_model.eval()

        input_ids, attention_mask, _ = self.get_dummy_inputs(device)

        lm_engine_output = lm_engine_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )
        bigcode_output = bigcode_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert lm_engine_output.sequences.equal(bigcode_output.sequences)

        for lm_engine_token_score, bigcode_token_score in zip(lm_engine_output.scores, bigcode_output.scores):
            self.assert_equal_tensors(
                lm_engine_token_score,
                bigcode_token_score,
                False,
                rtol_float32=0,
                atol_float32=3e-7,
                rtol_float16=1e-2,
                atol_float16=4e-4,
                rtol_bfloat16=5e-3,
                atol_bfloat16=5e-3,
            )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(),
            TestCommons.get_attention_head_types(),
            TestCommons.get_position_embedding_types(),
            TestCommons.get_dtypes(),
        )
    )
    def test_generation_works(
        self,
        device: torch.device,
        attention_head_type: str,
        position_embedding_type: str,
        torch_dtype: torch.dtype,
    ) -> None:
        self.skip_test_if_device_unavailable(device)
        self.skip_test_if_layernorm_kernel_unavailable(device, torch_dtype)

        for config in [
            self.get_dense_test_config(attention_head_type, position_embedding_type),
            self.get_moe_test_config(attention_head_type, position_embedding_type),
        ]:
            model = self.from_config(config, torch_dtype=torch_dtype).to(device)
            model.eval()

            input_ids, attention_mask, _ = self.get_dummy_inputs(device)

            model.generate(input_ids=input_ids, attention_mask=attention_mask)
