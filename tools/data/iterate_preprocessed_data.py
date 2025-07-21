# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset
from lm_engine.utils import log_rank_0, set_logger


set_logger()


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument("--path-prefix", type=str, required=True, help="Path to binary file without suffix")

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    dataset = MMapIndexedDataset(args.path_prefix)

    log_rank_0(logging.INFO, f"number of documents in the dataset = {len(dataset)}")

    tokens = 0
    for document in tqdm(dataset):
        tokens += len(document)

    log_rank_0(logging.INFO, f"number of tokens in the dataset = {tokens}")


if __name__ == "__main__":
    main()
