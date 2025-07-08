# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
import tempfile

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM

from lm_engine.hf_models import CommonConfig, export_to_huggingface, import_from_huggingface
from lm_engine.hf_models.config import _Mamba2Args

from ..test_common import TestCommons


class ModelConversionTest(TestCommons):
    def model_conversion_test(
        self,
        lm_engine_config: CommonConfig,
        model_type: str,
        device: torch.device,
        exact_match: bool = True,
        compare_loss: bool = True,
        logits_rtol_float32: float = 0,
        logits_atol_float32: float = 3e-7,
        logits_rtol_float16: float = 0,
        logits_atol_float16: float = 3e-7,
        logits_rtol_bfloat16: float = 0,
        logits_atol_bfloat16: float = 3e-7,
        loss_rtol_float32: float = 0,
        loss_atol_float32: float = 1e-5,
        loss_rtol_float16: float = 0,
        loss_atol_float16: float = 1e-5,
        loss_rtol_bfloat16: float = 0,
        loss_atol_bfloat16: float = 1e-5,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        lm_engine_model = AutoModelForCausalLM.from_config(lm_engine_config).to(device)
        lm_engine_model.eval()

        with tempfile.TemporaryDirectory() as tmp_path:
            save_path = os.path.join(tmp_path, "save")
            export_path = os.path.join(tmp_path, "export")
            import_path = os.path.join(tmp_path, "import")

            lm_engine_model.save_pretrained(save_path, safe_serialization=True)

            export_to_huggingface(save_path, export_path, model_type=model_type)
            import_from_huggingface(export_path, import_path)

            assert self.compare_saved_models(save_path, import_path)

            hf_model = AutoModelForCausalLM.from_pretrained(export_path).to(device)
            hf_model.eval()

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)

        hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        hf_logits = hf_output.logits
        hf_loss = hf_output.loss

        lm_engine_output = lm_engine_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )
        lm_engine_logits = lm_engine_output.logits
        lm_engine_loss = lm_engine_output.loss

        # we don't care about what happens on masked values (they don't match btw)
        hf_logits[attention_mask == 0] = 0
        lm_engine_logits[attention_mask == 0] = 0

        self.assert_equal_tensors(
            lm_engine_logits,
            hf_logits,
            exact_match,
            rtol_float32=logits_rtol_float32,
            atol_float32=logits_atol_float32,
            rtol_float16=logits_rtol_float16,
            atol_float16=logits_atol_float16,
            rtol_bfloat16=logits_rtol_bfloat16,
            atol_bfloat16=logits_atol_bfloat16,
        )

        if compare_loss:
            self.assert_equal_tensors(
                lm_engine_loss,
                hf_loss,
                exact_match,
                rtol_float32=loss_rtol_float32,
                atol_float32=loss_atol_float32,
                rtol_float16=loss_rtol_float16,
                atol_float16=loss_atol_float16,
                rtol_bfloat16=loss_rtol_bfloat16,
                atol_bfloat16=loss_atol_bfloat16,
            )

    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices(), ["mha", "mqa"]))
    def test_bigcode_model_conversion(self, device: torch.device, attention_head_type: str) -> None:
        lm_engine_config = self.get_dense_test_config(attention_head_type, "learned_absolute")

        self.model_conversion_test(
            lm_engine_config=lm_engine_config, model_type="gpt_bigcode", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(), TestCommons.get_attention_head_types(), [True, False]
        )
    )
    def test_llama_model_conversion(self, device: torch.device, attention_head_type: str, add_bias: bool) -> None:
        lm_engine_config = self.get_dense_test_config(
            attention_head_type,
            "rope",
            add_bias=add_bias,
            activation_function="swiglu",
            normalization_function="rmsnorm",
        )

        self.model_conversion_test(
            lm_engine_config=lm_engine_config, model_type="llama", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_all_devices(), TestCommons.get_attention_head_types(), [True, False]
        )
    )
    def test_granite_model_conversion(self, device: torch.device, attention_head_type: str, add_bias: bool) -> None:
        lm_engine_config = self.get_dense_test_config(
            attention_head_type,
            "rope",
            add_bias=add_bias,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        self.model_conversion_test(
            lm_engine_config=lm_engine_config, model_type="granite", device=device, exact_match=False
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_granitemoe_model_conversion(self, device: torch.device, attention_head_type: str) -> None:
        lm_engine_config = self.get_moe_test_config(
            attention_head_type,
            "rope",
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="granitemoe",
            device=device,
            exact_match=False,
            compare_loss=False,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_granitemoeshared_model_conversion(self, device: torch.device, attention_head_type: str) -> None:
        lm_engine_config = self.get_moe_test_config(
            attention_head_type,
            "rope",
            add_bias=False,
            shared_n_inner=64,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="granitemoeshared",
            device=device,
            exact_match=False,
            compare_loss=False,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_granitemoehybrid_model_conversion(self, device: torch.device, attention_head_type: str) -> None:
        lm_engine_config = self.get_moe_test_config(
            attention_head_type,
            "nope",
            add_bias=False,
            shared_n_inner=64,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

        for layer in range(lm_engine_config.num_layers):
            if layer % 2 == 0:
                lm_engine_config.sequence_mixer_blocks[layer] = _Mamba2Args(intermediate_size=256)

        self.model_conversion_test(
            lm_engine_config=lm_engine_config,
            model_type="granitemoehybrid",
            device=device,
            exact_match=False,
            compare_loss=False,
            logits_atol_float32=2.5e-5,
        )
