# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .arguments import UnshardingArgs, get_args
from .checkpointing import load_checkpoint_and_unshard
from .utils import ProcessGroupManager, run_rank_n


def main() -> None:
    """main program"""

    args = get_args(UnshardingArgs)

    model, _, state_dict = load_checkpoint_and_unshard(args, mode, allowed_meta_device=True)
    run_rank_n(model.save_pretrained, barrier=ProcessGroupManager.is_initialized())(
        args.unsharded_path, state_dict=state_dict
    )


if __name__ == "__main__":
    main()
