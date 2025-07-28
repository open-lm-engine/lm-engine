# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


def divide_if_divisible(dividend: int, divisor: int, msg: str) -> int:
    """divide if divisible else raise an error

    Args:
        dividend (int): dividend
        divisor (int): divisor
        msg (str): error message

    Returns:
        int: result
    """

    assert dividend % divisor == 0, msg
    return dividend // divisor
