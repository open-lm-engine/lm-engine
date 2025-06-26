# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .enums import Mode
from .pretrain import main


if __name__ == "__main__":
    main(Mode.distillation)
