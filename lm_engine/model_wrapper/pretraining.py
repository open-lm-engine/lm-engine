# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.distributed._tensor.placement_types import Replicate
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..dtensors import tensor_to_dtensor
from ..enums import Kernel, Mode
from ..hf_models import (
    CausalLMOutputWithPast,
    PipelineParallelInput,
    PipelineParallelOutput,
    get_autoregressive_language_modeling_loss,
    is_aux_loss_zero,
)
from ..kernels import is_kernel_allowed
from ..utils import MetricsTrackingDict, ProcessGroupManager
from .base import ModelWrapper
from .utils import broadcast_tensor_parallel_input


class ModelWrapperForPretraining(ModelWrapper):
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

        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.reset_attention_mask = reset_attention_mask
        self.reset_position_ids = reset_position_ids

        super().__init__(
            mode=mode,
            model_name=model_name,
            pretrained_config=pretrained_config,
            model_class=model_class,
            dtype=dtype,
            efficient_initialization=efficient_initialization,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            num_pipeline_stages=num_pipeline_stages,
            pipeline_stage_id=pipeline_stage_id,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
        )

        if self.is_pipeline_parallel_enabled:
            assert not self.reset_attention_mask, "reset_attention_mask is not supported with pipeline parallelism"
            assert not self.reset_position_ids, "reset_position_ids is not supported with pipeline parallelism"

            self._extra_metrics = MetricsTrackingDict({})

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

        if not self.is_custom_model:
            assert not is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

        if isinstance(batch, torch.Tensor):
            batch = {"text": batch}

        if self.is_pipeline_parallel_enabled:
            batch["aux_loss_from_pipeline_parallel"] = aux_loss_from_pipeline_parallel
        else:
            assert aux_loss_from_pipeline_parallel == 0

        batch = self._prepare_model_inputs(batch)
        labels = batch.pop("labels")
        output: CausalLMOutputWithPast | PipelineParallelOutput = self.model(**batch, return_dict=True)

        if self.is_pipeline_parallel_enabled:
            # aux_loss is returned as a 0 dimensional tensor
            aux_loss = output.aux_loss
            use_aux_loss = not is_aux_loss_zero(aux_loss)

            if use_aux_loss and aux_loss.dim() == 0:
                aux_loss = aux_loss.unsqueeze(0)

            if self.is_last_stage:
                assert isinstance(output, CausalLMOutputWithPast)
                output = output.logits
            else:
                assert isinstance(output, PipelineParallelOutput)
                output = output.hidden_states

            if use_aux_loss:
                output = (output, aux_loss)
        else:
            output = self.get_loss(output, labels, lm_loss_multiplier=lm_loss_multiplier)

        return output

    def get_loss(
        self, model_outputs: CausalLMOutputWithPast, labels: torch.Tensor, lm_loss_multiplier: float = 1
    ) -> torch.Tensor | dict:
        tensor_parallel_enabled = ProcessGroupManager.is_tensor_parallel_enabled()
        use_fused_linear_cross_entropy_kernel = is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

        lm_loss = get_autoregressive_language_modeling_loss(
            lm_logits=None if use_fused_linear_cross_entropy_kernel else model_outputs.logits,
            labels=labels,
            hidden_states=model_outputs.last_hidden_state if use_fused_linear_cross_entropy_kernel else None,
            vocab_weight=self.model.get_output_embeddings().weight if use_fused_linear_cross_entropy_kernel else None,
            cu_seqlens=None,
            use_padding_free_transformer=self.use_padding_free_transformer,
            reduction="sum",
            shift_logits_and_labels=False,
            tensor_parallel_enabled=tensor_parallel_enabled,
        )

        lm_loss = lm_loss * lm_loss_multiplier
        aux_loss = getattr(model_outputs, "aux_loss", 0)

        if is_aux_loss_zero(aux_loss):
            loss = lm_loss
            output = {"loss": loss, "lm_loss": loss}
        else:
            if self.is_pipeline_parallel_enabled:
                self._extra_metrics = self._extra_metrics + {"aux_loss": aux_loss}

            if tensor_parallel_enabled:
                aux_loss = tensor_to_dtensor(aux_loss, device_mesh=self.tp_mesh, current_placement=Replicate())

            loss = _F.apply(lm_loss, aux_loss, self.router_aux_loss_coef)
            output = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}

        return output

    def get_extra_metrics(self) -> dict:
        if "aux_loss" in self._extra_metrics:
            self._extra_metrics["aux_loss"] = self._extra_metrics["aux_loss"].squeeze(0)

        return self._extra_metrics

    def reset_extra_metrics(self) -> None:
        self._extra_metrics = MetricsTrackingDict({})

    def _prepare_model_inputs(self, batch: dict) -> dict:
        if self.is_pipeline_parallel_enabled:
            # when using pipeline parallel, we broadcast the input outside the model function
            tokens = batch["text"]
            aux_loss_from_pipeline_parallel = batch["aux_loss_from_pipeline_parallel"]

            tokens = tokens.to(torch.cuda.current_device())

            if self.is_first_stage:
                input_ids = tokens[:, :-1]
                pipeline_parallel_input = None
            else:
                input_ids = None
                pipeline_parallel_input = PipelineParallelInput(
                    hidden_states=tokens, aux_loss=aux_loss_from_pipeline_parallel
                )

            batch = {"labels": None, "pipeline_parallel_input": pipeline_parallel_input}
        else:
            if ProcessGroupManager.is_tensor_parallel_enabled():
                tokens = broadcast_tensor_parallel_input(
                    None if batch is None else batch["text"], (self.micro_batch_size, self.sequence_length + 1)
                )
            else:
                tokens = batch["text"]
                tokens = tokens.to(torch.cuda.current_device())

            input_ids = tokens[:, :-1]
            batch = {"labels": tokens[:, 1:]}

        if self.use_padding_free_transformer:
            batch_size, sequence_length = input_ids.shape
            input_ids = input_ids.reshape(-1)

            if self.reset_attention_mask:
                num_tokens_in_batch = batch_size * sequence_length

                document_end_positions = input_ids == self.eos_token_id
                for i in range(sequence_length - 1, num_tokens_in_batch, sequence_length):
                    document_end_positions[i] = 1
                cu_seqlens = document_end_positions.nonzero(as_tuple=True)[0] + 1
                cu_seqlens = torch.cat([torch.tensor([0], device=input_ids.device), cu_seqlens])
                cu_seqlens = cu_seqlens.to(torch.int32)

                seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
                # we move to CPU here otherwise FlashAttention will move to CPU on every invocation i.e all layers
                max_seqlen = seqlen.max().item()

                if self.reset_position_ids:
                    position_ids = torch.cat(
                        [torch.arange(0, i, 1, dtype=torch.int32, device=input_ids.device) for i in seqlen]
                    )
                else:
                    position_ids = self.position_ids
            else:
                cu_seqlens = self.cu_seqlens
                max_seqlen = self.sequence_length
                position_ids = self.position_ids

            batch["cu_seqlens"] = cu_seqlens
            batch["max_seqlen"] = max_seqlen
            batch["position_ids"] = position_ids

        batch["input_ids"] = input_ids

        if ProcessGroupManager.is_tensor_parallel_enabled():
            batch["output_parallel_lm_logits"] = True

        return batch

    def _setup_model(self) -> None:
        assert not self.is_encoder_decoder, "currently encoder_decoder models are not supported for pretraining"

        super()._setup_model()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_padding_free_transformer:
            if not self.reset_attention_mask:
                self.register_buffer(
                    "cu_seqlens",
                    torch.arange(
                        0,
                        self.micro_batch_size * self.sequence_length + 1,
                        self.sequence_length,
                        dtype=torch.int32,
                        device=torch.cuda.current_device(),
                    ),
                    persistent=False,
                )

            if self.reset_position_ids:
                assert self.reset_attention_mask, "reset_attention_mask should be specified with reset_position_ids"
            else:
                self.register_buffer(
                    "position_ids",
                    torch.arange(0, self.sequence_length, 1, device=torch.cuda.current_device()).repeat(
                        self.micro_batch_size
                    ),
                    persistent=False,
                )
        else:
            assert (
                not self.reset_attention_mask
            ), "currently reset_attention_mask is only implemented for padding free transformer"
            assert (
                not self.reset_position_ids
            ), "currently reset_position_ids is only implemented for padding free transformer"


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lm_loss: torch.Tensor, aux_loss: torch.Tensor, router_aux_loss_coef: float) -> torch.Tensor:
        ctx.router_aux_loss_coef = router_aux_loss_coef
        return lm_loss + router_aux_loss_coef * aux_loss

    @staticmethod
    @torch._dynamo.disable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        return grad_output, ctx.router_aux_loss_coef * grad_output, None
