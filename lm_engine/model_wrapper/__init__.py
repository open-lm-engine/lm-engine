# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ..arguments import DistillationArgs, InferenceArgs, TrainingArgs, UnshardingArgs
from ..containers import ModelContainer
from ..enums import Mode, TuningMethod
from ..kernels import enable_kernels
from ..utils import get_pipeline_stage_ids_on_current_rank
from .base import ModelWrapper
from .distillation import ModelWrapperForDistillation
from .finetuning import ModelWrapperForFinetuning
from .pretraining import ModelWrapperForPretraining
from .utils import broadcast_tensor_parallel_input


_MODEL_CLASS_MAPPING = {
    TuningMethod.pretraining: ModelWrapperForPretraining,
    TuningMethod.full_finetuning: ModelWrapperForFinetuning,
    TuningMethod.distillation: ModelWrapperForDistillation,
}


def get_model_container(
    args: TrainingArgs | InferenceArgs | UnshardingArgs | DistillationArgs, mode: Mode
) -> ModelContainer:
    tuning_method = args.tuning_args.tuning_method
    num_pipeline_stages = args.distributed_args.num_pipeline_stages

    if tuning_method != TuningMethod.pretraining:
        assert num_pipeline_stages == 1, "pipeline parallelism is only supported with pretraining"

    if tuning_method not in _MODEL_CLASS_MAPPING:
        raise ValueError(f"unexpected tuning_method ({tuning_method})")

    kwargs = {
        "mode": mode,
        "model_name": args.model_args.model_name,
        "pretrained_config": args.model_args.pretrained_config,
        "model_class": args.model_args.model_class,
        "dtype": args.mixed_precision_args.dtype,
        "efficient_initialization": args.model_args.efficient_initialization,
        "use_padding_free_transformer": args.model_args.use_padding_free_transformer,
        "sequence_parallel": args.distributed_args.sequence_parallel,
        "num_pipeline_stages": num_pipeline_stages,
        "trust_remote_code": args.model_args.trust_remote_code,
        "tokenizer_name": args.tokenizer_args.tokenizer_name,
        "additional_special_tokens": args.tokenizer_args.additional_special_tokens,
    }

    # pretraining model wrapper needs some extra arguments for initialization
    if tuning_method in [TuningMethod.pretraining, TuningMethod.distillation]:
        kwargs["micro_batch_size"] = args.training_parameters.micro_batch_size
        kwargs["sequence_length"] = args.datasets[0].class_args.get("sequence_length")
        kwargs["reset_attention_mask"] = args.model_args.reset_attention_mask
        kwargs["reset_position_ids"] = args.model_args.reset_position_ids

    if tuning_method == TuningMethod.distillation:
        kwargs["teacher_model_name"] = args.teacher_args.model_name
        kwargs["teacher_model_class"] = args.teacher_args.model_class
        kwargs["teacher_model_dtype"] = args.teacher_args.dtype
        kwargs["kl_divergence_method"] = args.teacher_args.kl_divergence_method
        kwargs["kl_divergence_weight"] = args.teacher_args.kl_divergence_weight

    # well, this is needed since HF models are kernel dependent and don't allow changing kernels on the fly
    with enable_kernels(args.kernel_args.kernels):
        model_list = []
        for pipeline_stage_id in get_pipeline_stage_ids_on_current_rank(num_pipeline_stages):
            kwargs["pipeline_stage_id"] = pipeline_stage_id
            model_list.append(_MODEL_CLASS_MAPPING[tuning_method](**kwargs))

    return ModelContainer(model_list)
