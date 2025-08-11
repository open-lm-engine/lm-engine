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

ANNEAL = False

MAX_ANNEALING_STEPS = 4096
ANNEALING_STEP = 0 if ANNEAL else 4096


def get_annealing_step():
    global ANNEALING_STEP
    return ANNEALING_STEP


def get_max_annealing_steps():
    global MAX_ANNEALING_STEPS
    return MAX_ANNEALING_STEPS


def update_annealing():
    global ANNEALING_STEP
    ANNEALING_STEP = min(ANNEALING_STEP + 1, MAX_ANNEALING_STEPS)
    # print("Updated to ", ANNEALING_STEP)


class ModelWrapperForPretrainingDiffusion(ModelWrapperForPretraining):
    def __init__(
        self,
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
        keep_in_fp32: bool = True,
    ) -> ModelWrapperForPretraining:
        super().__init__(
            model_name,
            pretrained_config,
            model_class,
            dtype,
            efficient_initialization,
            use_padding_free_transformer,
            sequence_parallel,
            micro_batch_size,
            sequence_length,
            num_pipeline_stages,
            pipeline_stage_id,
            trust_remote_code,
            tokenizer_name,
            additional_special_tokens,
            reset_attention_mask,
            reset_position_ids,
            keep_in_fp32,
        )
        assert self.use_padding_free_transformer and self.reset_attention_mask

    def _get_model_kwargs(self):
        kwargs = super()._get_model_kwargs()
        if hasattr(self, "mask_token_id"):
            kwargs["mask_token_id"] = self.mask_token_id
        return kwargs

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
        masked_indices = batch["masked_indices"]
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
            assert (labels[batch["masked_indices"]] != self.ignore_token_id).all()
            output = self.get_loss(output, labels, masked_indices, p_mask, lm_loss_multiplier=lm_loss_multiplier)

        return output

    def get_loss(
        self,
        model_outputs: CausalLMOutputWithPast,
        labels: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        lm_loss_multiplier: float = 1,
    ) -> torch.Tensor | dict:
        tensor_parallel_enabled = ProcessGroupManager.is_tensor_parallel_enabled()
        # use_fused_linear_cross_entropy_kernel = is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)
        flat_logits = model_outputs.logits.flatten(0, -2)
        flat_labels = labels.flatten()[masked_indices]
        flat_p_mask = p_mask.flatten()[masked_indices]
        # print(flat_logits.size(), flat_labels.size())
        lm_loss = (
            F.cross_entropy(
                input=flat_logits,
                target=flat_labels,
                ignore_index=self.ignore_token_id,
                reduction="none",
            )
            / flat_p_mask
        ).sum() / 2

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
        self.mask_token_id = 100351  # self.tokenizer.convert_tokens_to_ids(FIM_MIDDLE)
        assert self.mask_token_id is not None
        self.pad_token_id = self.tokenizer.pad_token_id
        assert self.pad_token_id is not None
        self.ignore_token_id = -1  # self.pad_token_id  # self.mask_token_id

    def _prepare_model_inputs(self, batch: dict) -> dict:
        device = torch.cuda.current_device()
        if self.is_pipeline_parallel_enabled:
            raise NotImplementedError("No pipeline for diffusion yet.")
        else:
            if ProcessGroupManager.is_tensor_parallel_enabled():
                tokens = broadcast_tensor_parallel_input(
                    None if batch is None else batch["text"], (self.micro_batch_size, self.sequence_length + 1)
                )
            else:
                tokens = batch["text"]
                tokens = tokens.to(device)
            # if torch.rand(1, device=tokens.device) < 0.5:
            #     unnoised_input_ids = tokens[:, 1:]
            # else:
            #     unnoised_input_ids = tokens[:, :-1]
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        orig_batch_size, sequence_length = input_ids.shape
        batch_size = orig_batch_size * 2

        perm_idxs = torch.argsort(torch.rand_like(input_ids[:, :-1], dtype=torch.bfloat16), dim=-1)
        # unnoised_input_ids = unnoised_input_ids.repeat_interleave(2, 0).flatten()
        # input_ids = unnoised_input_ids.clone()
        input_ids = input_ids.repeat_interleave(2, 0).flatten()
        orig_input_ids = input_ids.clone()
        unmasked_labels = labels.repeat_interleave(2, 0).flatten()
        labels = torch.full_like(input_ids, fill_value=self.ignore_token_id)
        p_mask = torch.ones_like(input_ids, dtype=torch.bfloat16)

        # assert batch_size % 2 == 0
        masked_ptr = 0
        masked_indices = (
            torch.zeros((batch_size // 2) * (sequence_length - 1), dtype=input_ids.dtype, device=input_ids.device) - 1
        )

        document_end_positions = unmasked_labels == self.eos_token_id
        document_end_positions[sequence_length - 1 :: sequence_length] = 1
        eps = 1e-4
        moved_boundary = False

        def _apply_mask_and_fill(start_idx, end_idx, masked_idxs, p):
            nonlocal moved_boundary
            # assert ((masked_idxs - 1) >= 0).all()
            labels[start_idx:end_idx][masked_idxs] = input_ids[start_idx:end_idx][masked_idxs + 1]
            input_ids[start_idx:end_idx][masked_idxs + 1] = self.mask_token_id
            p_mask[start_idx:end_idx] = p

            # prob = torch.rand(1, device=tokens.device)
            # if prob < 0.5:
            #     end_positions = unnoised_input_ids[start_idx:end_idx] == self.eos_token_id
            #     end_positions_noised = input_ids[start_idx:end_idx] == self.eos_token_id
            #     # find mismatches
            #     end_position_mismatch = (end_positions != end_positions_noised) & (
            #         input_ids[start_idx:end_idx] == self.mask_token_id
            #     )
            #     if end_position_mismatch.any():
            #         movable_locs = torch.nonzero(end_position_mismatch, as_tuple=True)[0]
            #         move_start_idx = movable_locs[torch.randint(movable_locs.size(0), size=(1,))[0]]
            #         rest_unmasked = input_ids[start_idx:end_idx][move_start_idx:] != self.mask_token_id
            #         if rest_unmasked.any():
            #             first_unmasked_idx = move_start_idx + torch.nonzero(rest_unmasked, as_tuple=True)[0].min()
            #             if first_unmasked_idx - move_start_idx > 1:
            #                 moved_boundary = True
            #                 document_end_positions[start_idx:end_idx][move_start_idx] = False
            #                 document_end_positions[start_idx:end_idx][first_unmasked_idx] = True
            #                 input_ids[start_idx:end_idx][first_unmasked_idx] = self.eos_token_id
            #                 labels[start_idx:end_idx][move_start_idx:first_unmasked_idx] = self.eos_token_id

        for i in range(orig_batch_size):
            t = torch.rand(1, device=input_ids.device)[0]
            p = (1 - 2 * eps) * t + eps
            sample_masked_idxs = perm_idxs[i]
            mask_count = torch.round(p * (sequence_length - 1)).to(torch.int32)
            masked_idxs_ = sample_masked_idxs[:mask_count]
            _apply_mask_and_fill(
                start_idx=2 * i * sequence_length, end_idx=(2 * i + 1) * sequence_length, masked_idxs=masked_idxs_, p=p
            )
            masked_indices[masked_ptr : masked_ptr + mask_count] = 2 * i * sequence_length + masked_idxs_
            masked_ptr += mask_count

            masked_idxs_ = sample_masked_idxs[mask_count:]
            mask_count = (sequence_length - 1) - mask_count
            _apply_mask_and_fill(
                start_idx=(2 * i + 1) * sequence_length,
                end_idx=(2 * i + 2) * sequence_length,
                masked_idxs=masked_idxs_,
                p=1 - p,
            )
            masked_indices[masked_ptr : masked_ptr + mask_count] = (2 * i + 1) * sequence_length + masked_idxs_
            masked_ptr += mask_count
        # assert (masked_indices != -1).any()

        masked_indices, _ = torch.sort(masked_indices)
        cu_seqlens = document_end_positions.nonzero(as_tuple=True)[0] + 1
        cu_seqlens = torch.cat([torch.tensor([0], device=input_ids.device), cu_seqlens]).to(torch.int32)
        seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
        # we move to CPU here otherwise FlashAttention will move to CPU on every invocation i.e all layers
        max_seqlen = seqlen.max().item()

        if self.reset_position_ids:
            position_ids = torch.cat(
                [torch.arange(0, i, 1, dtype=torch.int32, device=input_ids.device) for i in seqlen]
            )
        else:
            position_ids = self.position_ids

        # masked_idxs = (labels != self.ignore_token_id).nonzero(as_tuple=True)[0]
        # masked_idxs, _ = torch.sort(masked_idxs)
        # print(labels[masked_indices], masked_indices)
        assert (labels[masked_indices] != self.ignore_token_id).all()
        assert (input_ids[masked_indices + 1] == self.mask_token_id).all()
        anneal_ratio = min((get_annealing_step() / get_max_annealing_steps()) / 0.5, 1)
        if ANNEAL:
            batch = {
                # "input_ids": input_ids,
                # "input_ids": orig_input_ids,
                "input_ids": torch.where(
                    torch.rand_like(orig_input_ids, dtype=torch.bfloat16) < anneal_ratio, input_ids, orig_input_ids
                ),
                "labels": unmasked_labels,
                # "labels": labels.flatten(),
                "p_mask": p_mask.flatten(),
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "position_ids": position_ids,
                "masked_indices": masked_indices,
            }
            update_annealing()
        else:
            batch = {
                "input_ids": input_ids,
                "labels": labels.flatten(),
                "p_mask": p_mask.flatten(),
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "position_ids": position_ids,
                "masked_indices": masked_indices,
            }

        # if True:
        #     from transformers import PreTrainedTokenizer
        #     tokenizer: PreTrainedTokenizer = self.tokenizer
        #     def to_token_list(seq):
        #         output = []
        #         for idx in seq:
        #             if idx == self.ignore_token_id:
        #                 c = "<I>"
        #             elif idx == self.mask_token_id:
        #                 c = "_"
        #             else:
        #                 c = tokenizer._convert_id_to_token(idx)
        #             output.append(c)
        #         return output
        #     print((input_ids == self.mask_token_id).int().sum().item())
        #     combine_seq = batch["input_ids"][1:].clone()
        #     mask = combine_seq == self.mask_token_id
        #     combine_seq[mask] = batch["labels"][:-1][mask]
        #     combine_seq = torch.cat([input_ids[:1], combine_seq], dim=0)

        #     for i in range(cu_seqlens.size(0) - 1):
        #         # seq_in = batch["input_ids"][cu_seqlens[i] : cu_seqlens[i + 1]]
        #         # seq_out = batch["labels"][cu_seqlens[i] : cu_seqlens[i + 1]]
        #         # seq = torch.where(seq_out == self.ignore_token_id, seq_in, seq_out)
        #         seq = combine_seq[cu_seqlens[i] : cu_seqlens[i + 1]]
        #         assert p_mask[cu_seqlens[i]] == p_mask[cu_seqlens[i + 1] - 1]
        #         print()
        #         print(cu_seqlens[i].item(), cu_seqlens[i + 1].item(), p_mask[cu_seqlens[i + 1] - 1].item())
        #         print(repr(tokenizer.convert_tokens_to_string(to_token_list(seq))))
        #         # print(repr(tokenizer.convert_tokens_to_string(to_token_list(seq_out))))
        #     print(cu_seqlens)
        #     exit()
        # else:
        #     print("No deletions.")

        if ProcessGroupManager.is_tensor_parallel_enabled():
            batch["output_parallel_lm_logits"] = True

        return batch
