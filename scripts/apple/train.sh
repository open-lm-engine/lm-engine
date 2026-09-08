TOKENIZERS_PARALLELISM=false torchrun -m lm_engine.training.train --config ${1}
