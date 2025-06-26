# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math
from typing import Iterable

import torch

from ..enums import LossMask, Mode
from ..hf_models import convert_padding_free_lists_to_tensors


def collate_fn(
    batch: list[dict],
    mode: Mode,
    loss_mask: LossMask,
    eos_token_id: int,
    is_encoder_decoder: bool,
    use_padding_free_transformer: bool,
    labels_mask_value: int = -100,
    pad_to_multiple_of: int = 1,
    device: torch.device = None,
) -> dict:
    """prepares the batch with padding to pass into the forward function of the HuggingFace model

    Args:
        batch (list[dict]): input tokens and output tokens. Output tokens are optional when running generation but required for training.

    Returns:
        dict: dict containing input_ids, attention_mask and labels if outputs is specified
    """

    inputs = [i["input"] for i in batch]
    outputs = [i["output"] for i in batch] if mode == Mode.training else None

    # labels is None when outputs is None
    labels = None

    device = torch.cuda.current_device() if device is None else device

    if use_padding_free_transformer:
        if is_encoder_decoder:
            raise NotImplementedError("padding free transformer only supports decoder only models")
        else:
            input_ids = inputs
            attention_mask = None

            if loss_mask == LossMask.output_only:
                labels = [
                    [labels_mask_value] * (len(array_in) - len(array_out)) + array_out
                    for array_in, array_out in zip(inputs, outputs)
                ]
            elif loss_mask == LossMask.no_mask:
                labels = inputs
            else:
                raise ValueError(f"unexpected loss_mask ({loss_mask})")

            tokens_to_add = 0
            if pad_to_multiple_of > 1:
                total_tokens = sum([len(array) for array in input_ids])
                tokens_to_add = (math.ceil(total_tokens / pad_to_multiple_of) * pad_to_multiple_of) - total_tokens

            # we pad the last example in the batch on the right
            # NOTE this can be done since the attention is causal
            input_ids[-1].extend([eos_token_id] * tokens_to_add)
            labels[-1].extend([labels_mask_value] * tokens_to_add)

            input_ids, position_ids, _, labels, cu_seqlens, max_seqlen = convert_padding_free_lists_to_tensors(
                input_ids=input_ids, labels=labels, device=device
            )

        result = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        if labels is not None:
            result["labels"] = labels
    else:
        if is_encoder_decoder:
            if pad_to_multiple_of > 1:
                raise NotImplementedError("pad_to_multiple_of is not implemented for encoder-decoder models")

            input_max_length = max(list(map(len, inputs)))

            input_ids = [[eos_token_id] * (input_max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (input_max_length - len(array)) + [1] * len(array) for array in inputs]

            if outputs is not None:
                assert (
                    loss_mask == LossMask.output_only
                ), "only output_only loss mask is supported with encoder decoder models"

                output_max_length = max(list(map(len, outputs)))
                # right padding for labels
                labels = [array + [labels_mask_value] * (output_max_length - len(array)) for array in outputs]
        else:
            max_length = max(list(map(len, inputs)))
            if pad_to_multiple_of > 1:
                max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of

            input_ids = [[eos_token_id] * (max_length - len(array)) + array for array in inputs]
            attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in inputs]

            if outputs is not None:
                if loss_mask == LossMask.output_only:
                    labels = [[labels_mask_value] * (max_length - len(array)) + array for array in outputs]
                elif loss_mask == LossMask.no_mask:
                    labels = inputs
                else:
                    raise ValueError(f"unexpected loss_mask ({loss_mask})")

        result = {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attention_mask, device=device),
        }
        if labels is not None:
            result["labels"] = torch.tensor(labels, device=device)

    return result


def custom_iterator(x: Iterable | None, infinite: bool) -> Iterable:
    """converts and iterable into a non-ending infinite iterable, will return None if input is None

    Args:
        x (Iterable): the iterable to convert
        infinite (bool): whether to return an infinite iterator

    Returns:
        Iterable: the converted iterable

    Yields:
        Iterator[Iterable]: an element from the original iterator
    """

    if x is None:
        return None

    def infinite_iterator(q):
        while True:
            for i in q:
                yield i

    iterator_function = infinite_iterator if infinite else iter
    return iterator_function(x)


def get_next_batch(x: Iterable | None) -> dict:
    """get next batch

    Args:
        x (Iterable): dataloader

    Returns:
        dict: batch
    """

    # train_dataloader is always None on TP ranks other than 0
    if x is None:
        return None

    return next(x)
