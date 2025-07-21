# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


def pad(
    inputs: list[list[int]], pad_token_id: int, pad_to_multiple_of: int = 1
) -> tuple[list[list[int]], list[list[int]]]:
    max_length = max(list(map(len, inputs)))

    if pad_to_multiple_of > 1:
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    input_ids = [[pad_token_id] * (max_length - len(array)) + array for array in inputs]
    attention_mask = [[0] * (max_length - len(array)) + [1] * len(array) for array in inputs]

    return input_ids, attention_mask
