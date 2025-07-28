# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoTokenizer

from lm_engine.arguments import DatasetArgs, DistributedArgs, RandomArgs, TrainingParameters
from lm_engine.data import get_megatron_gpt_dataloaders
from lm_engine.utils import ProcessGroupManager, set_logger


# this is needed to print logs during preparation of dataloader
set_logger()


# use the appropriate tokenizer here
# the tokenizer is required for using FIM during training
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base")

# adjust these hyperparameters
num_training_steps = 10000
# batch size per GPU
micro_batch_size = 8
sequence_length = 2048
# whether or not to evaluate during training
eval_during_training = False
# interval at which eval is done if eval_during_training is True
eval_interval = 1000


class DummyArgs:
    # seed for shuffling data
    random_args = RandomArgs(seed=42)

    datasets = [
        DatasetArgs(
            # don't change these values
            class_name="MegatronDataset",
            data_name="Megatron",
            data_sampling_ratio=1,
            # you are allowed to change these values
            class_args={
                # number of steps to eval every time we do evaluation
                # 2 means (2 * micro_batch_size * sequence_length) tokens to calculate perplexity
                "eval_steps": 2,
                # path of cache used by megatron dataset
                "data_cache_path": "./cache",
                # sampling proportion followed by their file paths
                "data_path": [
                    0.2,
                    "/proj-3500/datasets/slim_pajama_gptneox_megatron/train/chunk1",
                    0.5,
                    "/proj-3500/datasets/slim_pajama_gptneox_megatron/train/chunk2",
                    0.3,
                    "/proj-3500/datasets/slim_pajama_gptneox_megatron/train/chunk3",
                ],
                # 100% train, 0% val and 0% test
                "split": "100,0,0",
                # megatron dataloader returns 2049 tokens (sequence_length + 1) since the loss computation is done outside
                # the model class. to train with HF's loss computation, we need to pass (sequence_length - 1) so it
                # returns 2048 tokens but note that HF will itself train with 2047 tokens which is not a big problem
                # Don't change this directly if using HF trainer or something adjust sequence_length above
                "sequence_length": sequence_length - 1,
            },
        )
    ]

    training_parameters = TrainingParameters(
        num_training_steps=num_training_steps,
        micro_batch_size=micro_batch_size,
        eval_interval=eval_interval,
        eval_during_training=eval_during_training,
    )

    # don't play with this value
    distributed_args = DistributedArgs()


assert (
    ProcessGroupManager.is_initialized()
), "torch distributed needs to be initialized before the following logic otherwise everything is incorrect"

# following logic is for FSDP/ZeRO without any model parallelism
# since the function is model parallelism aware, we need to emulate no model parallelism
# if you are using model parallelism, either train directly on lm-engine or try modifying this logic
with (
    ProcessGroupManager.set_dummy_data_parallel_world_size(ProcessGroupManager.get_world_size()),
    ProcessGroupManager.set_dummy_data_parallel_rank(ProcessGroupManager.get_global_rank()),
    ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
    ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
    ProcessGroupManager.set_dummy_tensor_parallel_first_rank(ProcessGroupManager.get_global_rank()),
):
    # running this function outside of the above context might throw error or give incorrect behaviour
    # you can change consumed samples, 0 means dataloader is starting from beginning and adjusting to a different
    # number can be used to resume training at a different point
    # for FSDP, consumed_samples = global_step * micro_batch_size * gradient_accumulation_steps * num_GPUs
    train_dataloader, val_dataloaders, test_dataloaders = get_megatron_gpt_dataloaders(
        DummyArgs(), tokenizer, consumed_samples=0
    )


# filter any dataloaders in val and test that are None
# this happens when there are no elements in these dataloaders
val_dataloaders = filter(lambda x: x is not None, val_dataloaders)
test_dataloaders = filter(lambda x: x is not None, test_dataloaders)


# use the dataloaders :)
