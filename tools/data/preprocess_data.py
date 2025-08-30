# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer

from lm_engine.data.megatron.preprocess_data import convert_file


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON/Arrow")
    group.add_argument(
        "--subset", type=str, default=None, help="Subset argument when loading input data from a HuggingFace dataset"
    )
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, required=True, help="Number of worker processes to launch")
    group.add_argument("--chunk-size", type=int, required=True, help="Chunk size assigned to each worker process")
    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    convert_file(
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
        input_file=args.input,
        output_prefix=args.output_prefix,
        workers=args.workers,
        chunk_size=args.chunk_size,
        subset=args.subset,
        json_keys=args.json_keys,
        append_eos_token=args.append_eod,
    )


if __name__ == "__main__":
    main()
