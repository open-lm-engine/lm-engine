# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..enums import Kernel
from ..hf_models import get_model_parallel_class, is_custom_model
from ..kernels import is_kernel_allowed
from ..tokenizers import get_tokenizer
from ..utils import ProcessGroupManager, SafeTensorsWeightsManager, log_rank_0, string_to_torch_dtype


class ModelWrapper(nn.Module):
    """Model class which wraps any HuggingFace model"""

    def __init__(
        self,
        model_name: str | None,
        pretrained_config: dict | None,
        dtype: torch.dtype,
        efficient_initialization: bool,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        num_pipeline_stages: int,
        pipeline_stage_id: int,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        keep_in_fp32: bool = True,
    ) -> ModelWrapper:
        """initializes a model wrapper for a HuggingFace model

        Args:
            model_name (str | None): path of the model on disk or HF hub
            pretrained_config (dict | None): config of the model to load model from, only used if `model_name` is None
            dtype (torch.dtype): dtype for the model
            efficient_initialization (bool): whether to use efficient initialization for the model initialization, saves CPU memory
            use_padding_free_transformer (bool): whether to use padding free transformer
            sequence_parallel (bool): whether to use sequence parallel
            num_pipeline_stages (int): number of stages for the pipeline
            pipeline_stage_id (int): current pipeline stage id
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            keep_in_fp32 (bool, optional): whether to keep model in fp32 right now. Defaults to True.
        """

        super().__init__()

        self.model_name = model_name
        self.pretrained_config = pretrained_config
        self.efficient_initialization = efficient_initialization
        self.dtype = dtype
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel
        self.tokenizer_name = self.model_name if tokenizer_name is None else tokenizer_name
        self.trust_remote_code = trust_remote_code
        self.keep_in_fp32 = keep_in_fp32

        self.num_pipeline_stages = num_pipeline_stages
        self.pipeline_stage_id = pipeline_stage_id
        self.is_first_stage = self.pipeline_stage_id == 0
        self.is_last_stage = self.pipeline_stage_id == self.num_pipeline_stages - 1
        self.is_pipeline_parallel_enabled = self.num_pipeline_stages > 1

        use_model_parallelism = ProcessGroupManager.is_tensor_parallel_enabled() or self.is_pipeline_parallel_enabled

        self._setup_config()
        self.is_custom_model = is_custom_model(self.config.model_type)

        total_parameters, active_parameters = self.calculate_num_parameters()

        log_rank_0(logging.INFO, f"num parameters in the model = {total_parameters:,}")
        log_rank_0(logging.INFO, f"active parameters in the model = {active_parameters:,}")

        if use_model_parallelism:
            self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
            self.model_class = get_model_parallel_class(self.config.model_type)
        else:
            self.model_class = AutoModelForCausalLM

        if self.use_padding_free_transformer:
            assert self.is_custom_model, "padding free transformer is not supported with the specified model"

        self._setup_tokenizer()
        self._setup_model()

        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            original_vocab_size = len(self.tokenizer)

            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            log_rank_0(logging.INFO, f"added {len(additional_special_tokens)} tokens")

            if len(self.tokenizer) != original_vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))

    def save_pretrained(self, save_path: str, state_dict: dict | None = None) -> None:
        self.tokenizer.save_pretrained(save_path, legacy_format=False)

        if state_dict is None:
            self.model.save_pretrained(save_path)
        else:
            for key in list(state_dict.keys()):
                assert key.startswith("model.")
                state_dict[_remove_first_occurance(key, "model.")] = state_dict.pop(key)

            self.config.save_pretrained(save_path)
            SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

    def _setup_config(self) -> None:
        self.config = (
            AutoConfig.for_model(**self.pretrained_config)
            if self.model_name is None
            else AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        )

        assert not self.config.is_encoder_decoder, "we don't support encoder-decoder models"

        self.tie_word_embeddings = self.config.tie_word_embeddings
        self.router_aux_loss_coef = getattr(self.config, "router_aux_loss_coef", None)

        log_rank_0(logging.INFO, self.config)

    def _setup_tokenizer(self) -> None:
        assert self.tokenizer_name is not None, "pass a tokenizer"

        self.tokenizer = get_tokenizer(AutoTokenizer.__name__, self.tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

    def _get_model_kwargs(self) -> dict:
        if self.model_name is None:
            model_kwargs = {"config": self.config}
        else:
            model_kwargs = {"pretrained_model_name_or_path": self.model_name}

        if not self.is_custom_model:
            assert not is_kernel_allowed(Kernel.flash_attention_3)
            model_kwargs["attn_implementation"] = (
                "flash_attention_2" if is_kernel_allowed(Kernel.flash_attention_2) else "sdpa"
            )

        if self.use_padding_free_transformer:
            model_kwargs["use_padding_free_transformer"] = True
        if self.sequence_parallel:
            model_kwargs["sequence_parallel"] = True
        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if self.is_pipeline_parallel_enabled:
            model_kwargs["num_pipeline_stages"] = self.num_pipeline_stages
            model_kwargs["pipeline_stage_id"] = self.pipeline_stage_id

        return model_kwargs

    def _setup_model(self) -> None:
        model_kwargs = self._get_model_kwargs()

        if self.model_name is None:
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id == self.config.bos_token_id

            if self.tokenizer.eos_token_id is not None:
                assert self.tokenizer.eos_token_id == self.config.eos_token_id

            if self.tokenizer.pad_token_id is not None:
                assert self.tokenizer.pad_token_id == self.config.pad_token_id

        context = nullcontext()
        kwargs = {}

        if self.keep_in_fp32:
            if self.efficient_initialization:
                if self.model_name is None:
                    context = torch.device("meta")
                else:
                    assert (
                        not ProcessGroupManager.is_tensor_parallel_enabled()
                    ), "tensor parallel models don't support efficient init with model name"

                    if ProcessGroupManager.get_data_parallel_rank() != 0:
                        context = torch.device("meta")
        elif self.dtype == "fp8":
            log_rank_0(logging.WARN, "dtype fp8 was passed but loading model in fp16")
            kwargs = {"dtype": torch.float16}
        else:
            kwargs = {"dtype": string_to_torch_dtype(self.dtype)}

        with context:
            if self.model_name is None:
                if self.is_pipeline_parallel_enabled or ProcessGroupManager.is_tensor_parallel_enabled():
                    # avoid inferring the model class so use _from_config instead of from_config
                    self.model = self.model_class._from_config(**model_kwargs, **kwargs)
                else:
                    self.model = self.model_class.from_config(**model_kwargs, **kwargs)
            else:
                self.model = self.model_class.from_pretrained(**model_kwargs, **kwargs)

    def calculate_num_parameters(self) -> tuple[int, int]:
        model_kwargs = self._get_model_kwargs()

        with torch.device("meta"):
            if self.model_name is not None:
                model_kwargs["config"] = AutoConfig.from_pretrained(model_kwargs.pop("pretrained_model_name_or_path"))

            model: nn.Module = AutoModelForCausalLM.from_config(**model_kwargs)

            num_parameters = 0
            for param in model.parameters():
                num_parameters += param.numel()

            active_parameters = 0

            def _recurse_immediate_children_and_count_active_parameters(module: nn.Module) -> None:
                nonlocal active_parameters

                for m in module.children():
                    if hasattr(m, "get_num_active_parameters"):
                        active_parameters += m.get_num_active_parameters()
                    else:
                        for parameter in m.parameters(recurse=False):
                            active_parameters += parameter.numel()

                        _recurse_immediate_children_and_count_active_parameters(m)

            _recurse_immediate_children_and_count_active_parameters(model)

            return num_parameters, active_parameters

    def has_teacher_model(self) -> bool:
        return False


def _remove_first_occurance(string: str, substring: str) -> str:
    if string.startswith(substring):
        string = string[len(substring) :]

    return string
