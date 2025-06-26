# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..dtensors import tensor_to_dtensor
from ..enums import Kernel, KLDivergenceMethod, Mode
from ..hf_models import (
    CausalLMOutputWithPast,
    PipelineParallelOutput,
    get_autoregressive_language_modeling_loss,
    is_aux_loss_zero,
)
from ..kernels import is_kernel_allowed
from ..utils import MetricsTrackingDict, ProcessGroupManager, log_rank_0, string_to_torch_dtype
from .base import ModelWrapper
from .pretraining import ModelWrapperForPretraining
from .utils import broadcast_tensor_parallel_input


class ModelWrapperForDistillation(ModelWrapperForPretraining):
    def __init__(
        self,
        mode: Mode,
        model_name: str | None,
        pretrained_config: dict | None,
        model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        dtype: torch.dtype,
        efficient_initialization: bool,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        micro_batch_size: int,
        sequence_length: int,
        num_pipeline_stages: int,
        pipeline_stage_id: int,
        teacher_model_name: str | None,
        teacher_model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        teacher_model_dtype: torch.dtype,
        kl_divergence_method: KLDivergenceMethod,
        kl_divergence_weight: float = 1,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        reset_attention_mask: bool = False,
        reset_position_ids: bool = False,
    ) -> None:
        """initializes a model wrapper for a HuggingFace model

        Args:
            mode (Mode): training / inference mode
            model_name (str | None): path of the model on disk or HF hub
            pretrained_config (dict | None): config of the model to load model from, only used if `model_name` is None
            model_class (AutoModelForCausalLM | AutoModelForSeq2SeqLM): HF model class to use for model loading
            dtype (torch.dtype): dtype for the model
            efficient_initialization (bool): whether to use efficient initialization for the model initialization, saves CPU memory
            use_padding_free_transformer (bool): whether to use padding free transformer
            sequence_parallel (bool): whether to use sequence parallel
            micro_batch_size (int): micro batch size for pretraining
            sequence_length (int): sequence length for pretraining
            num_pipeline_stages (int): number of stages for the pipeline
            pipeline_stage_id (int): current pipeline stage id
            trust_remote_code (bool, optional): whether the model has remote code in the HF bucket. Defaults to False.
            tokenizer_name (str | None, optional): path of the model on disk or HF hub. Defaults to None. If None, the `model_name` is used for tokenizer.
            additional_special_tokens (list[str] | None, optional): additional special tokens to use for expanding tokenizer. Defaults to None.
            reset_attention_mask (bool, optional): whether to reset attention mask during pretraining. Defaults to False.
            reset_position_ids (bool, optional): whether to reset position ids during pretraining. Defaults to False.
        """

        self.teacher_model_class = teacher_model_class
        self.teacher_model_name = teacher_model_name
        self.teacher_model_dtype = teacher_model_dtype
        self.kl_divergence_method = kl_divergence_method
        self.kl_divergence_weight = kl_divergence_weight

        super().__init__(
            mode=mode,
            model_name=model_name,
            pretrained_config=pretrained_config,
            model_class=model_class,
            dtype=dtype,
            efficient_initialization=efficient_initialization,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
            num_pipeline_stages=num_pipeline_stages,
            pipeline_stage_id=pipeline_stage_id,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
            reset_attention_mask=reset_attention_mask,
            reset_position_ids=reset_position_ids,
        )

        if ProcessGroupManager.is_tensor_parallel_enabled() or num_pipeline_stages > 1:
            raise NotImplementedError()

    def forward(
        self,
        batch: dict | torch.Tensor,
        aux_loss_from_pipeline_parallel: torch.Tensor | float = 0,
        lm_loss_multiplier: float = 1,
    ) -> dict:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        # for pretraining we compute loss externally here instead of relying on transformers.
        # this is done because megatron's dataset returns batches of length (sequence_length + 1)
        # instead of (sequence_length), so we need to trim the input_ids before forward pass.
        # transformers does forward pass before however and then trims the tokens.

        batch = self._prepare_model_inputs(batch)
        labels = batch.pop("labels")
        output: CausalLMOutputWithPast | PipelineParallelOutput = self.model(**batch, return_dict=True)

        assert not is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

        student_logits = output.logits
        del output

        # TODO modify this when TP support is added
        lm_loss = get_autoregressive_language_modeling_loss(
            lm_logits=student_logits,
            labels=labels,
            hidden_states=None,
            vocab_weight=None,
            cu_seqlens=None,
            use_padding_free_transformer=self.use_padding_free_transformer,
            reduction="sum",
            shift_logits_and_labels=False,
            tensor_parallel_enabled=False,
        )

        lm_loss = lm_loss * lm_loss_multiplier

        with torch.inference_mode():
            output: CausalLMOutputWithPast | PipelineParallelOutput = self.teacher_model(**batch, return_dict=True)
            teacher_logits = output.logits
            teacher_logits = teacher_logits.float()

        teacher_log_softmax = F.log_softmax(teacher_logits, dim=-1).view(-1, teacher_logits.size(-1))
        student_log_softmax = F.log_softmax(student_logits, dim=-1).view(-1, student_logits.size(-1))

        if self.kl_divergence_method == KLDivergenceMethod.forward:
            # sum [student * ln(student / teacher)]
            kl_divergence = F.kl_div(teacher_log_softmax, student_log_softmax, reduction="batchmean", log_target=True)
        elif self.kl_divergence_method == KLDivergenceMethod.backward:
            # sum [teacher * ln(teacher / student)]
            kl_divergence = F.kl_div(student_log_softmax, teacher_log_softmax, reduction="batchmean", log_target=True)

        loss = lm_loss + self.kl_divergence_weight * kl_divergence

        return {"loss": loss, "lm_loss": lm_loss, "kl_divergence": kl_divergence}

    def _setup_config(self) -> None:
        super()._setup_config()

        self.teacher_config = AutoConfig.from_pretrained(
            self.teacher_model_name, trust_remote_code=self.trust_remote_code
        )

    def _setup_tokenizer(self) -> None:
        super()._setup_tokenizer()

        log_rank_0(
            logging.WARN,
            "tokenizers should be same for both teacher and student, unfortunately I don't know how to check for this",
        )

    def _setup_model(self) -> None:
        super()._setup_model()

        self.teacher_model = self.teacher_model_class.from_pretrained(
            self.teacher_model_name, torch_dtype=string_to_torch_dtype(self.teacher_model_dtype)
        )
        self.teacher_model.eval()

    def has_teacher_model(self) -> bool:
        return True

    def train(self, mode: bool = True):
        super().train(mode)
        # teacher model should always be in eval mode
        self.teacher_model.eval()
        return self
