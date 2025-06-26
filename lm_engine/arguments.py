# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from argparse import ArgumentParser
from typing import Any

import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .defaults import INPUT_FORMAT, OUTPUT_FORMAT
from .enums import (
    ExperimentsTrackerName,
    GradientCheckpointingMethod,
    Kernel,
    KLDivergenceMethod,
    LossMask,
    LRDecaySchedule,
    Mode,
    ParamsGroupMethod,
    TuningMethod,
)
from .utils import BaseArgs, load_yaml, log_environment, log_rank_0, normalize_dtype_string, run_rank_n, set_logger


def _check_not_None(object_name_list: list[tuple[Any, str]]) -> None:
    for obj, name in object_name_list:
        assert obj is not None, f"{name} cannot be None"


class RandomArgs(BaseArgs):
    # random seed
    seed: int = 42


class TokenizerArgs(BaseArgs):
    # override model's tokenizer with this
    tokenizer_name: str | None = None
    # add special tokens to the tokenizer
    additional_special_tokens: list[str] | None = None


class ModelArgs(BaseArgs):
    # model name on huggingface hub
    model_name: str | None = None
    # config class to load the model from
    pretrained_config: dict | None = None
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # trust remote code for models that are not directly supported by HuggingFace yet
    trust_remote_code: bool = False
    # whether to use padding free transformer: https://huggingface.co/blog/mayank-mishra/padding-free-transformer
    use_padding_free_transformer: bool = False
    # use lower memory to initialize model
    efficient_initialization: bool = False
    # whether to reset attention masks for pretraining
    reset_attention_mask: bool = False
    # whether to reset position ids for pretraining
    reset_position_ids: bool = False

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.model_class, "model_class")])

        # model_name
        if self.model_name is None:
            _check_not_None([(self.pretrained_config, "pretrained_config")])
        else:
            assert self.pretrained_config is None, "pretrained_config shouldn't be specified with model_name"

        assert self.model_class in [
            AutoModelForCausalLM.__name__,
            AutoModelForSeq2SeqLM.__name__,
        ], f"unexpected model_class ({self.model_class})"

        self.model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM = getattr(transformers, self.model_class)


class TuningArgs(BaseArgs):
    # type of tuning, full finetuning / pretraining / distillation
    tuning_method: TuningMethod = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.tuning_method, "tuning_method")])


class TrainingParameters(BaseArgs):
    # whether to use sequential sampler for validation
    ignore_sampling_proportion_for_validation: bool = False
    # number of training steps
    num_training_steps: int | None = None
    # gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # interval for evaluation
    eval_interval: int | None = None
    # batch size per GPU for ZeRO-DP
    micro_batch_size: int = None
    # whether to use val dataset for validation during training
    eval_during_training: bool = True
    # masking methodology of loss function input
    loss_mask: LossMask = LossMask.output_only
    # gradient clip value
    gradient_clipping: float | None = 1

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.num_training_steps, "num_training_steps"), (self.micro_batch_size, "micro_batch_size")])

        # eval_interval
        if self.eval_during_training:
            _check_not_None([(self.eval_interval, "eval_interval")])


class SaveArgs(BaseArgs):
    # path to save checkpoints
    save_path: str = None
    # interval for checkpointing
    save_interval: int = None
    # whether to save optimizer
    save_optimizer: bool = True
    # whether to use async checkpointing
    async_checkpointing: bool = False

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.save_path, "save_path"), (self.save_interval, "save_interval")])


class LoadArgs(BaseArgs):
    # path to load checkpoints
    load_path: str = None
    # iteration to load
    iteration: int | None = None
    # whether to load optimizer
    load_optimizer: bool = True
    # whether to load lr_scheduler
    load_lr_scheduler: bool = True
    # whether to load rng state
    load_rng_state: bool = True
    # whether to resume dataloader
    load_dataloader_state: bool = True
    # whether to resume experiments tracker
    load_experiments_tracker_state: bool = True
    # whether to load starting iteration
    load_starting_iteration: bool = True
    # whether to resume learning rate during training
    # this is a NO-OP if we are loading LR scheduler
    resume_learning_rate: bool = True

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.load_path, "load_path")])

        if not self.load_optimizer:
            assert (
                not self.load_lr_scheduler
            ), "lr_scheduler loading doesn't make sense if you aren't loading optimizer"

        if self.load_lr_scheduler:
            assert self.resume_learning_rate, "resume learning rate needs to be True when reloading LR scheduler"


class DatasetArgs(BaseArgs):
    # dataset class
    class_name: str = None
    # class args for dataset
    class_args: dict = {}
    # dataset name
    data_name: str = None
    # formatting to use for input
    input_format: str = INPUT_FORMAT
    # formatting to use for output
    output_format: str = OUTPUT_FORMAT
    # data sampling proportions
    data_sampling_ratio: int = None
    # max tokens for input text
    max_input_tokens: int | None = None
    # max tokens for output text
    max_output_tokens: int | None = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.class_name, "dataset class_name"), (self.data_name, "data_name")])

        # data_sampling_ratios
        if self.data_sampling_ratio is not None:
            assert self.data_sampling_ratio > 0, "data_sampling_ratio should be a positive integer"


class OptimizerArgs(BaseArgs):
    # optimizer class
    class_name: str = "TorchAdamW"
    # how to create param groups
    params_group_method: ParamsGroupMethod | None = None
    # backward hooked optimizer
    use_optimizer_with_backward_hook: bool = False
    # class args for optimizer
    class_args: dict = {
        "lr": 1e-5,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-10,
    }

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.class_name, "optimizer class_name")])


class LRSchedulerArgs(BaseArgs):
    # warmup steps
    num_warmup_steps: int = 200
    # constant steps after warmup and before decay
    num_constant_steps: int = 0
    # decays steps after constant steps, if None then all remaining steps are for decay
    num_decay_steps: int | None = None
    # lr scheduler for decay
    lr_decay_style: LRDecaySchedule = LRDecaySchedule.cosine
    # decay factor * max_lr = min_lr (ratio of min_lr and max_lr)
    lr_decay_factor: float = 0.1
    # coefficients to use in advanced LR schedules, including power
    # {"a": batch_size, "b": -0.51, "c": batch_size * sequence_length}
    extra_lr_scheduler_args: dict = {}


class MixedPrecisionArgs(BaseArgs):
    # dtype to use for training / inference
    dtype: str = "fp32"
    # fp8
    scaling_type_input: str = "dynamic"
    scaling_type_weight: str = "dynamic"
    scaling_type_grad_output: str = "dynamic"

    def model_post_init(self, __context: Any) -> None:
        # dtype
        self.dtype = normalize_dtype_string(self.dtype)


class ZeroTopologyArgs(BaseArgs):
    # GPUs to use for replication
    data_parallel_replication_world_size: int | None = None
    # GPUs to use for sharding
    data_parallel_sharding_world_size: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.data_parallel_replication_world_size is None:
            assert (
                self.data_parallel_sharding_world_size is None
            ), "data_parallel_replication_world_size needs to be specified with data_parallel_sharding_world_size"
        else:
            assert (
                self.data_parallel_sharding_world_size is not None
            ), "data_parallel_sharding_world_size needs to be specified with data_parallel_replication_world_size"


class DistributedArgs(BaseArgs):
    # ZeRO stage
    stage: int = 3
    # train with CPU offloading to save GPU memory
    cpu_offload: bool = False
    # whether to use gradient checkpointing, enabling leads to lower memory usage with increased step time
    gradient_checkpointing_method: GradientCheckpointingMethod | None = None
    # gradient checkpointint args
    gradient_checkpointing_args: dict = {}
    # zero topology
    zero_topology: ZeroTopologyArgs = ZeroTopologyArgs()
    # communication dtype
    communication_dtype: str | None = None
    # whether to use torch.compile
    torch_compile: bool = False
    # whether to use a dispatching dataloader
    dispatching_dataloader: bool = False
    # tensor parallel world size
    tensor_parallel_world_size: int = 1
    # whether to use sequence parallel
    sequence_parallel: bool = False
    # pipeline parallel world size
    pipeline_parallel_world_size: int = 1
    # distributed timeout for NCCL in minutes
    timeout_minutes: int | None = None
    # fsdp algorithm
    fsdp_algorithm: int = 2
    # whether to sync every gradient accumulation step
    sync_every_gradient_accumulation_step: bool = False
    # total number of pipeline stages
    num_pipeline_stages: int = 1
    # pipeline parallel shedule to use
    pipeline_parallel_schedule: str | None = None
    # whether to use async-TP
    use_async_tensor_parallel: bool = False

    def model_post_init(self, __context: Any) -> None:
        # communication dtype
        if self.communication_dtype is not None:
            self.communication_dtype = normalize_dtype_string(self.communication_dtype)

        if self.sequence_parallel:
            assert self.tensor_parallel_world_size > 1, "tensor parallel needs to be enabled for sequence parallel"

        if self.tensor_parallel_world_size > 1:
            assert self.fsdp_algorithm == 2, "FSDP-2 is required for using tensor parallel"

        if self.use_async_tensor_parallel:
            assert self.sequence_parallel, "sequence parallel should be enabled for using async-TP"

        assert (
            self.num_pipeline_stages % self.pipeline_parallel_world_size == 0
        ), "num_pipeline_stages should be a multiple of pipeline_parallel_world_size"

        if self.num_pipeline_stages > 1:
            _check_not_None([(self.pipeline_parallel_schedule, "pipeline_parallel_schedule")])


class AimArgs(BaseArgs):
    # aim repo, experiment logs are saved here
    repo: str = None
    # name of the experiment
    experiment: str = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.repo, "repo"), (self.experiment, "experiment")])


class WandBArgs(BaseArgs):
    # aim repo, experiment logs are saved here
    project: str = None
    # name of the experiment
    name: str = None
    # run hash for the experiment
    entity: str | None = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.project, "project"), (self.name, "name")])


class LoggingArgs(BaseArgs):
    # logging level
    logging_level: str = "INFO"
    # log interval
    log_interval: int = 10
    # arguments if using aim
    aim_args: AimArgs | None = None
    # arguments if using wandb
    wandb_args: WandBArgs | None = None
    # experiment tracker to use (aim or wandb)
    experiments_tracker_name: ExperimentsTrackerName | None = None
    # whether to use colored logs
    use_colored_logs: bool = False
    # torch profiler trace path, specifying a path will enable the torch profiler
    # this can cause some performance impact so use sparingly
    torch_profiler_trace_path: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.experiments_tracker_name == ExperimentsTrackerName.aim:
            _check_not_None([(self.aim_args, "aim_args")])
        elif self.experiments_tracker_name == ExperimentsTrackerName.wandb:
            _check_not_None([(self.wandb_args, "wandb_args")])


class KernelArgs(BaseArgs):
    # custom kernels
    kernels: list[Kernel] = []


class TeacherArgs(BaseArgs):
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # model name on huggingface hub
    model_name: str | None = None
    # teacher dtype
    dtype: str = "fp32"
    # KL divergence method
    kl_divergence_method: KLDivergenceMethod = None
    # KL divergence weight
    kl_divergence_weight: float = 1

    def model_post_init(self, __context: Any) -> None:
        # dtype
        self.dtype = normalize_dtype_string(self.dtype)

        assert self.model_class in [
            AutoModelForCausalLM.__name__,
            AutoModelForSeq2SeqLM.__name__,
        ], f"unexpected model_class ({self.model_class})"

        self.model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM = getattr(transformers, self.model_class)

        _check_not_None([(self.kl_divergence_method, "kl_divergence_method")])


class TrainingArgs(BaseArgs):
    # randomization related arguments
    random_args: RandomArgs = RandomArgs()
    # tokenizer related arguments
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    # model related arguments
    model_args: ModelArgs = None
    # tuning related arguments
    tuning_args: TuningArgs = None
    # optimizer related arguments
    optimizer_args: OptimizerArgs = OptimizerArgs()
    # lr_scheduler related arguments
    lr_scheduler_args: LRSchedulerArgs = LRSchedulerArgs()
    # list of datasets to use
    datasets: list[DatasetArgs] = []
    # save related arguments
    save_args: SaveArgs = None
    # load related arguments
    load_args: LoadArgs | None = None
    # training parameters
    training_parameters: TrainingParameters | None = None
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # mixed precision related arguments
    mixed_precision_args: MixedPrecisionArgs = MixedPrecisionArgs()
    # distributed training related arguments
    distributed_args: DistributedArgs = DistributedArgs()
    # kernel args
    kernel_args: KernelArgs = KernelArgs()

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.model_args, "model_args"),
                (self.tuning_args, "tuning_args"),
                (self.save_args, "save_args"),
                (self.datasets, "datasets"),
            ]
        )

        # datasets
        _check_datasets(self.datasets)

        if self.distributed_args.num_pipeline_stages > 1 and self.training_parameters.eval_during_training:
            raise NotImplementedError("evaluation is not supported with pipeline parallel")

        if self.optimizer_args.use_optimizer_with_backward_hook:
            assert self.training_parameters.gradient_accumulation_steps == 1
            assert self.training_parameters.gradient_clipping is None

            raise NotImplementedError(
                "use_optimizer_with_backward_hook doesn't support saving or loading checkpoint, comment this "
                "assertion out to play with this, this is purely experimental"
            )


class GenerationParameters(BaseArgs):
    # batch size
    batch_size: int = None
    # sample or greedy
    do_sample: bool | None = None
    # max new tokens to generate
    max_new_tokens: int = None
    # temperature
    temperature: float | None = None
    # top k
    top_k: int | None = None
    # top p
    top_p: float | None = None

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.batch_size, "batch_size"), (self.max_new_tokens, "max_new_tokens")])


class InferenceArgs(BaseArgs):
    # randomization related arguments
    random_args: RandomArgs = RandomArgs()
    # tokenizer related arguments
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    # model related arguments
    model_args: ModelArgs | None = None
    # list of datasets to use
    datasets: list[DatasetArgs] = []
    # load related arguments
    load_args: LoadArgs | None = None
    # generation parameters
    generation_parameters: GenerationParameters = None
    # mixed precision related arguments
    mixed_precision_args: MixedPrecisionArgs = MixedPrecisionArgs()
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # output dir
    output_dir: str = None
    # kernel args
    kernel_args: KernelArgs = KernelArgs()

    def model_post_init(self, __context: Any) -> None:
        _check_not_None(
            [
                (self.datasets, "datasets"),
                (self.generation_parameters, "generation_parameters"),
                (self.output_dir, "output_dir"),
            ]
        )

        if self.load_args is None:
            assert self.model_args is not None, "model_args need to be specified if load_args are not specified"
        else:
            assert self.model_args is None, "model_args can't be specified with load_args"

        # datasets
        _check_datasets(self.datasets)


class UnshardingArgs(BaseArgs):
    # load related arguments
    load_args: LoadArgs = None
    # unsharded path
    unsharded_path: str = None
    # mixed precision related arguments
    mixed_precision_args: MixedPrecisionArgs = MixedPrecisionArgs()
    # logging related arguments
    logging_args: LoggingArgs = LoggingArgs()
    # kernel args
    kernel_args: KernelArgs = KernelArgs()

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.load_args, "load_args"), (self.unsharded_path, "unsharded_path")])


class DistillationArgs(TrainingArgs):
    # teacher model arguments
    teacher_args: TeacherArgs = None
    # kernel args
    kernel_args: KernelArgs = KernelArgs()

    def model_post_init(self, __context: Any) -> None:
        _check_not_None([(self.teacher_args, "teacher_args")])

        super().model_post_init(__context)


_MODE_ARGS_MAP = {
    Mode.training: TrainingArgs,
    Mode.inference: InferenceArgs,
    Mode.unsharding: UnshardingArgs,
    Mode.distillation: DistillationArgs,
}


def get_args(mode: Mode) -> TrainingArgs | InferenceArgs | UnshardingArgs:
    """get args for training / inference

    Args:
        mode (Mode): training / inference mode for running the program

    Returns:
        TrainingArgs | InferenceArgs | UnshardingArgs: args for training / inference
    """

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path for the config")
    args = parser.parse_args()

    config: dict = load_yaml(args.config)
    args: TrainingArgs | InferenceArgs | UnshardingArgs = _MODE_ARGS_MAP[mode](**config)

    set_logger(args.logging_args.logging_level, colored_log=args.logging_args.use_colored_logs)
    log_args(args)
    log_environment()

    return args


@run_rank_n
def log_args(args: TrainingArgs | InferenceArgs | UnshardingArgs) -> None:
    """log args

    Args:
        args (Union[TrainingArgs, InferenceArgs, UnshardingArgs]): args for training / inference
    """

    def _iterate_args_recursively(
        args: TrainingArgs | InferenceArgs | UnshardingArgs | dict | BaseArgs, prefix: str = ""
    ) -> None:
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


def _check_datasets(datasets: list[DatasetArgs]) -> None:
    assert len(datasets) != 0, "datasets cannot be an empty list"
    # check data_names are unique
    assert len(datasets) == len(
        set([dataset.data_name for dataset in datasets])
    ), "data_name should be unique for each dataset"
