# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.distributed
from torch.distributed._tensor.placement_types import Replicate

from ..communication import Communication
from ..dtensors import tensor_to_dtensor
from ..enums import Kernel
from ..hf_models import CausalLMOutputWithPast, get_autoregressive_language_modeling_loss
from ..kernels import is_kernel_allowed
from ..utils import MetricsTrackingDict, ProcessGroupManager
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    def forward(self, batch: dict, lm_loss_multiplier: float = 1) -> MetricsTrackingDict:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            MetricsTrackingDict: loss tracking dict
        """

        if ProcessGroupManager.is_tensor_parallel_enabled():
            batch = self._broadcast_inputs_for_tensor_parallel(batch)

        if not self.is_custom_model:
            assert not is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

        labels = batch.pop("labels")
        model_outputs: CausalLMOutputWithPast = self.model(**batch)

        return self.get_loss(
            model_outputs=model_outputs,
            labels=labels,
            cu_seqlens=batch.get("cu_seqlens", None),
            lm_loss_multiplier=lm_loss_multiplier,
        )

    def get_loss(
        self,
        model_outputs: CausalLMOutputWithPast,
        labels: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        lm_loss_multiplier: float = 1,
    ) -> torch.Tensor | dict:
        tensor_parallel_enabled = ProcessGroupManager.is_tensor_parallel_enabled()
        use_fused_linear_cross_entropy_kernel = is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

        lm_loss = get_autoregressive_language_modeling_loss(
            lm_logits=None if use_fused_linear_cross_entropy_kernel else model_outputs.logits,
            labels=labels,
            hidden_states=model_outputs.last_hidden_state if use_fused_linear_cross_entropy_kernel else None,
            vocab_weight=self.model.get_output_embeddings().weight if use_fused_linear_cross_entropy_kernel else None,
            cu_seqlens=cu_seqlens,
            use_padding_free_transformer=self.use_padding_free_transformer,
            reduction="sum",
            shift_logits_and_labels=True,
            tensor_parallel_enabled=tensor_parallel_enabled,
        )

        lm_loss = lm_loss * lm_loss_multiplier
        aux_loss = getattr(model_outputs, "aux_loss", 0)

        if aux_loss == 0:
            loss = lm_loss
            output = {"loss": loss}
        else:
            if tensor_parallel_enabled:
                aux_loss = tensor_to_dtensor(aux_loss, device_mesh=self.tp_mesh, current_placement=Replicate())

            loss = lm_loss + self.router_aux_loss_coef * aux_loss
            output = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}

        return output

    def _broadcast_inputs_for_tensor_parallel(self, batch: dict) -> dict:
        device = torch.cuda.current_device()

        is_tp_first_rank = ProcessGroupManager.is_tensor_parallel_first_rank()
        tp_source_rank = ProcessGroupManager.get_tensor_parallel_first_rank()
        tp_group = ProcessGroupManager.get_tensor_parallel_group()

        if self.use_padding_free_transformer:
            keys = ["input_ids", "position_ids", "labels", "cu_seqlens", "max_seqlen"]

            if is_tp_first_rank:
                metadata = torch.tensor([batch["cu_seqlens"].numel(), batch["input_ids"].numel()], device=device)
            else:
                metadata = torch.empty(2, dtype=torch.long, device=device)

            torch.distributed.broadcast(metadata, src=tp_source_rank, group=tp_group)
            cu_seqlens_num_elements, input_ids_num_elements = metadata

            if not is_tp_first_rank:
                batch = {
                    "input_ids": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "position_ids": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "labels": torch.empty(input_ids_num_elements, dtype=torch.long, device=device),
                    "cu_seqlens": torch.empty(cu_seqlens_num_elements, dtype=torch.int32, device=device),
                    "max_seqlen": torch.empty(1, dtype=torch.long, device=device),
                }
        else:
            keys = ["input_ids", "attention_mask", "labels"]

            batch_shape = batch["input_ids"].shape if is_tp_first_rank else None
            batch_shape = Communication.broadcast_object(batch_shape, src=tp_source_rank, group=tp_group)

            if not is_tp_first_rank:
                batch = {key: torch.empty(batch_shape, dtype=torch.long, device=device) for key in keys}

        for key in keys:
            torch.distributed.broadcast(batch[key], src=tp_source_rank, group=tp_group)

        return batch
