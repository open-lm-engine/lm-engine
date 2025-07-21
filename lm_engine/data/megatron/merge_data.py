from lm_engine.data.megatron.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def merge_files(input_prefixes: list[str], output_prefix: str) -> None:
    builder = MMapIndexedDatasetBuilder(
        get_bin_path(output_prefix), dtype=MMapIndexedDataset(input_prefixes[0]).index.dtype
    )

    for input_prefix in input_prefixes:
        builder.add_index(input_prefix)

    builder.finalize(get_idx_path(output_prefix))
