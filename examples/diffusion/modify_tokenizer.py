import sys
from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_args() -> Namespace:
    parser = ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-path", type=str, required=True, help="Path to binary output file without suffix")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=4096)
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.model_max_length = sys.maxsize
    print("bos_token_id", tokenizer.bos_token_id)
    print("eos_token_id", tokenizer.eos_token_id)
    print("pad_token_id", tokenizer.pad_token_id)
    print("Vocab size:", len(tokenizer))
    tokenizer.save_pretrained(args.output_path)
