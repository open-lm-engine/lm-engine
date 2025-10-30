# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from importlib.metadata import distributions
from warnings import warn

from .parallel import ProcessGroupManager, is_tracking_rank, run_rank_n


_LOGGER: logging.Logger = None


def set_logger(level: int = logging.INFO, colored_log: bool = False) -> None:
    stream = logging.StreamHandler()

    if colored_log:
        from .packages import is_colorlog_available

        assert is_colorlog_available(), "pip package colorlog is needed for colored logging"
        from colorlog import ColoredFormatter

        stream.setFormatter(ColoredFormatter("%(asctime)s - %(log_color)s[%(levelname)-8s] ▶%(reset)s %(message)s"))
        logging.basicConfig(level=level, handlers=[stream], force=True)
    else:
        logging.basicConfig(
            level=level, handlers=[stream], format="%(asctime)s - [%(levelname)-8s] ▶ %(message)s", force=True
        )

    global _LOGGER
    _LOGGER = logging.getLogger()


def get_logger() -> logging.Logger:
    return _LOGGER


@run_rank_n
def log_rank_0(level: int, msg: str) -> None:
    logger = get_logger()

    if logger is None:
        set_logger()
        log_rank_0(logging.WARN, "logger is not initialized yet, initializing now")
    else:
        logger.log(level=level, msg=msg, stacklevel=3)


def log_metrics(level: int, msg: str) -> None:
    if not is_tracking_rank():
        return

    get_logger().log(level=level, msg=msg, stacklevel=3)


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    """print on a single process"""

    print(*args, **kwargs)


def print_ranks_all(*args, **kwargs) -> None:
    """print on all processes sequentially, blocks other process and is slow. Please us sparingly."""

    for rank in range(ProcessGroupManager.get_world_size()):
        run_rank_n(print, rank=rank, barrier=True)(f"rank {rank}:", *args, **kwargs)


@run_rank_n
def warn_rank_0(*args, **kwargs) -> None:
    """warn on a single process"""

    warn(*args, **kwargs, stacklevel=3)


@run_rank_n
def log_args(args: TrainingArgs | UnshardingArgs) -> None:
    """log args

    Args:
        args (Union[TrainingArgs, UnshardingArgs]): args for training / inference
    """

    def _iterate_args_recursively(args: TrainingArgs | UnshardingArgs | dict | BaseArgs, prefix: str = "") -> None:
        result = []

        if isinstance(args, BaseArgs):
            args = vars(args)

        p = len(prefix)

        for k, v in args.items():
            suffix = "." * (48 - len(k) - p)

            if isinstance(v, (BaseArgs, dict)):
                if isinstance(v, dict) and len(v) == 0:
                    result.append(f"{prefix}{k} {suffix} " + r"{}")
                else:
                    kv_list_subargs = _iterate_args_recursively(v, prefix + " " * 4)
                    result.append(f"{prefix}{k}:\n" + "\n".join(kv_list_subargs))
            elif isinstance(v, list) and all([isinstance(v_, (BaseArgs, dict)) for v_ in v]):
                kv_list_subargs = []
                for v_ in v:
                    v_ = _iterate_args_recursively(v_, prefix + " " * 4)
                    kv_list_subargs.append(f"\n".join(v_))
                result.append(f"{prefix}{k}:\n" + ("\n" + " " * (p + 4) + "*" * (44 - p) + "\n").join(kv_list_subargs))
            else:
                result.append(f"{prefix}{k} {suffix} " + str(v))

        result.sort(key=lambda x: x.lower())
        return result

    log_rank_0(logging.INFO, "------------------------ arguments ------------------------")
    for line in _iterate_args_recursively(args):
        line = line.split("\n")
        for l in line:
            log_rank_0(logging.INFO, l)
    log_rank_0(logging.INFO, "-------------------- end of arguments ---------------------")


@run_rank_n
def log_environment() -> None:
    packages = sorted(["{}=={}".format(d.metadata["Name"], d.version) for d in distributions()])

    log_rank_0(logging.INFO, "------------------------ packages ------------------------")
    for package in packages:
        log_rank_0(logging.INFO, package)
    log_rank_0(logging.INFO, "-------------------- end of packages ---------------------")
