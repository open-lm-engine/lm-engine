# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math
from typing import Iterable

import torch

from ..enums import LossMask


def _check_list_type(list_of_list: list[list[int | float]] | None, error_message: str) -> None:
    if list_of_list is None:
        return

    assert isinstance(list_of_list, list), error_message
    assert isinstance(list_of_list[0], list), error_message


def _flatten_and_convert_to_tensors(x: list[int], device: torch.device) -> torch.Tensor:
    y = []
    for sequence in x:
        y.extend(sequence)

    return torch.tensor(y, device=device)


def convert_padding_free_lists_to_tensors(
    input_ids: list[list[int]] | None = None,
    position_ids: list[list[int]] | None = None,
    labels: list[list[int]] | None = None,
    device: torch.device = None,
) -> tuple[torch.Tensor | int]:

    # check input types are correct
    error_message = "{variable} should be of type List[List[{dtype}]]"
    _check_list_type(input_ids, error_message.format(variable="input_ids", dtype="int"))
    _check_list_type(position_ids, error_message.format(variable="position_ids", dtype="int"))
    _check_list_type(labels, error_message.format(variable="labels", dtype="int"))

    # prepare inputs for the model
    seqlens = torch.tensor([0] + [len(x) for x in input_ids], device=device)
    cu_seqlens = seqlens.cumsum(dim=-1).to(torch.int32)
    max_seqlen = seqlens.max().item()

    if position_ids is None:
        position_ids = [list(range(len(x))) for x in input_ids]
    position_ids = _flatten_and_convert_to_tensors(position_ids, device)

    input_ids = _flatten_and_convert_to_tensors(input_ids, device)

    if labels is not None:
        labels = _flatten_and_convert_to_tensors(labels, device)

    return input_ids, position_ids, labels, cu_seqlens, max_seqlen


def collate_fn(
    batch: list[dict],
    use_output: bool,
    loss_mask: LossMask,
    eos_token_id: int,
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
    outputs = [i["output"] for i in batch] if use_output else None

    # labels is None when outputs is None
    labels = None

    device = torch.cuda.current_device() if device is None else device

    input_ids = inputs

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
