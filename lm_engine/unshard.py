# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .arguments import get_args
from .checkpointing import load_checkpoint_for_inference
from .enums import Mode
from .utils import ProcessGroupManager, run_rank_n


def main() -> None:
    """main program"""

    mode = Mode.unsharding

    args = get_args(mode)

    model, _, state_dict = load_checkpoint_for_inference(args, mode, allowed_meta_device=True)
    run_rank_n(model.save_pretrained, barrier=ProcessGroupManager.is_initialized())(
        args.unsharded_path, state_dict=state_dict
    )


if __name__ == "__main__":
    main()
