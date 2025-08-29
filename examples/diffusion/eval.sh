#!/bin/bash

MODEL_PATH="$1"
set -x
RESULT_PATH=$MODEL_PATH/results/
mkdir -p $RESULT_PATH
export PYTHONPATH=./cute-kernels
# accelerate launch diffusion_eval.py --tasks wikitext \
accelerate launch diffusion_eval.py --tasks wikitext \
    --model lm_engine_diffusion --batch_size 8 \
    --model_args pretrained=${MODEL_PATH},mc_num=128 | tee $RESULT_PATH/wikitext.log
exit
accelerate launch diffusion_eval.py --tasks hellaswag \
    --num_fewshot 0 --model llada_dist --batch_size 8 \
    --model_args model_path=${MODEL_PATH},cfg=0.5,is_check_greedy=False,mc_num=128 | tee $RESULT_PATH/hellaswag.log
accelerate launch diffusion_eval.py --tasks winogrande \
    --num_fewshot 5 --model llada_dist --batch_size 8 \
    --model_args model_path=${MODEL_PATH},cfg=0.0,is_check_greedy=False,mc_num=128 | tee $RESULT_PATH/winogrande.log
accelerate launch diffusion_eval.py --tasks arc_challenge \
    --num_fewshot 0 --model llada_dist --batch_size 8 \
    --model_args model_path=${MODEL_PATH},cfg=0.5,is_check_greedy=False,mc_num=128 | tee $RESULT_PATH/arc_challenge.log

accelerate launch diffusion_eval.py --tasks arc_easy \
    --num_fewshot 0 --model llada_dist --batch_size 8 \
    --model_args model_path=${MODEL_PATH},cfg=0.5,is_check_greedy=False,mc_num=128  | tee $RESULT_PATH/arc_easy.log

accelerate launch diffusion_eval.py --tasks mmlu     --num_fewshot 5 --model llada_dist --batch_size 1 \
    --model_args model_path=${MODEL_PATH},cfg=0.0,is_check_greedy=False,mc_num=1 | tee $RESULT_PATH/mmlu.log

