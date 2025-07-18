# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._tensor.placement_types import Replicate
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from ..dtensors import tensor_to_dtensor
from ..enums import Kernel
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
from .pretraining import _F, ModelWrapperForPretraining
from .utils import broadcast_tensor_parallel_input


FIM_MIDDLE = "<fim_middle>"


class ModelWrapperForPretrainingDiffusion(ModelWrapperForPretraining):

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
        p_mask = batch.pop("p_mask")
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
            output = self.get_loss(output, labels, p_mask, lm_loss_multiplier=lm_loss_multiplier)

        return output

    def get_loss(
        self,
        model_outputs: CausalLMOutputWithPast,
        labels: torch.Tensor,
        p_mask: torch.Tensor,
        lm_loss_multiplier: float = 1,
    ) -> torch.Tensor | dict:
        tensor_parallel_enabled = ProcessGroupManager.is_tensor_parallel_enabled()
        # use_fused_linear_cross_entropy_kernel = is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)
        flat_logits = model_outputs.logits.flatten(0, -2)
        flat_labels = labels.flatten()
        # print(flat_logits.size(), flat_labels.size())
        lm_loss = (
            F.cross_entropy(
                input=flat_logits,
                target=flat_labels,
                ignore_index=self.mask_token_id,
                reduction="none",
            )
            / p_mask.flatten()
        ).sum()

        # lm_loss = get_autoregressive_language_modeling_loss(
        #     lm_logits=None if use_fused_linear_cross_entropy_kernel else model_outputs.logits,
        #     labels=labels,
        #     hidden_states=model_outputs.last_hidden_state if use_fused_linear_cross_entropy_kernel else None,
        #     vocab_weight=self.model.get_output_embeddings().weight if use_fused_linear_cross_entropy_kernel else None,
        #     cu_seqlens=None,
        #     use_padding_free_transformer=self.use_padding_free_transformer,
        #     reduction="sum",
        #     shift_logits_and_labels=False,
        #     tensor_parallel_enabled=tensor_parallel_enabled,
        # )

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

    def _setup_tokenizer(self) -> None:
        super()._setup_tokenizer()
        # TODO (shawntan) Use FIM token for now. Figure out if there is a way to have actual mask token.
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(FIM_MIDDLE)
        assert self.mask_token_id is not None

    def _forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        labels = torch.where(~masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, labels, p_mask

    def _prepare_model_inputs(self, batch: dict) -> dict:
        if self.is_pipeline_parallel_enabled:
            raise NotImplementedError("No pipeline for diffusion yet.")
        #     # when using pipeline parallel, we broadcast the input outside the model function
        #     tokens = batch["text"]
        #     aux_loss_from_pipeline_parallel = batch["aux_loss_from_pipeline_parallel"]

        #     tokens = tokens.to(torch.cuda.current_device())

        #     if self.is_first_stage:
        #         input_ids = tokens
        #         pipeline_parallel_input = None
        #     else:
        #         input_ids = None
        #         pipeline_parallel_input = PipelineParallelInput(
        #             hidden_states=tokens, aux_loss=aux_loss_from_pipeline_parallel
        #         )

        #     batch = {"labels": None, "pipeline_parallel_input": pipeline_parallel_input}
        else:
            if ProcessGroupManager.is_tensor_parallel_enabled():
                tokens = broadcast_tensor_parallel_input(
                    None if batch is None else batch["text"], (self.micro_batch_size, self.sequence_length + 1)
                )
            else:
                tokens = batch["text"]
                tokens = tokens.to(torch.cuda.current_device())

            unnoised_input_ids = tokens[:, 1:]
            input_ids, labels, p_mask = self._forward_process(unnoised_input_ids)
            batch = {"labels": labels, "p_mask": p_mask}

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
