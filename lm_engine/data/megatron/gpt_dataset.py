# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time

import numpy
from transformers import AutoTokenizer

from ...utils import log_rank_0
from .blended_megatron_dataset_config import GPTDatasetConfig
from .indexed_dataset import MMapIndexedDataset
from .megatron_dataset import MegatronDataset
from .utils import Split, build_sample_idx


FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_PAD = "<fim_pad>"


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The GPT-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        tokenizer: AutoTokenizer,
        config: GPTDatasetConfig,
        caching_allowed: bool,
    ) -> None:
        super().__init__(indexed_dataset, indexed_indices, num_samples, index_split, config, caching_allowed)

        self.tokenizer = tokenizer

        self.fim_rate = config.fim_rate
        self.fim_spm_rate = config.fim_spm_rate
        self.np_rng = numpy.random.RandomState(seed=config.random_seed)  # rng state for FIM

        if self.fim_rate != 0:
            self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (
                self.tokenizer.convert_tokens_to_ids(tok) for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
            )
            self.eos_token_id = self.tokenizer.eos_token_id
            assert self.eos_token_id is not None


    def _finalize(self) -> None:
        """Abstract method implementation

        Load or build/cache the document, sample, and shuffle indices
        """
        assert isinstance(self.config, GPTDatasetConfig)

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: int) -> dict[str, numpy.ndarray]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The text ids and (optionally) the document ids wrapped in a
            dictionary
        """
        text, document_ids = self._query_document_sample_shuffle_indices(idx)
        if getattr(self.config, "return_document_ids"):
            return {"text": text, "document_ids": document_ids}
        else:
            return {"text": text}

    @staticmethod
    def is_multimodal() -> bool:
        """Abstract method implementation

        Returns:
            bool: False
        """
        return False

    @staticmethod
    def is_split_by_sequence() -> bool:
        """Abstract method implementation

        Returns:
            bool: True
        """
        return True

    def _query_document_sample_shuffle_indices(self, idx: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.indexed_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(self.indexed_dataset.get(self.document_index[i], offset=offset, length=length))
        sample = numpy.array(numpy.concatenate(sample_parts), dtype=numpy.int64)

        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.

        if self.fim_rate != 0:
            sample_len = sample.shape[0]

            assert self.fim_rate <= 1 and self.fim_rate >= 0, "FIM rate must be a probability 0 <= rate <= 1"

            # eod = self.tokenizer.eod
            eod = self.eos_token_id
            segment_breaks = numpy.argwhere(sample == eod)  # split sample by document

            if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
                curr_start_position = 0
                new_samples = []
                for loc in numpy.nditer(segment_breaks):
                    # Only permute non-empty segments.
                    if loc - curr_start_position > 0:
                        # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                        permuted, self.np_rng = permute(
                            sample[curr_start_position:loc],
                            self.np_rng,
                            self.fim_rate,
                            self.fim_spm_rate,
                            self.tokenizer,
                            truncate_or_pad=False,
                            suffix_tok_id=self.suffix_tok_id,
                            prefix_tok_id=self.prefix_tok_id,
                            middle_tok_id=self.middle_tok_id,
                            pad_tok_id=self.pad_tok_id,
                        )
                        new_samples += [permuted, [eod]]

                    curr_start_position = loc + 1  # jump over the EOD token
                # Permute the segment after the last EOD
                permuted, self.np_rng = permute(
                    sample[curr_start_position:],
                    self.np_rng,
                    self.fim_rate,
                    self.fim_spm_rate,
                    self.tokenizer,
                    truncate_or_pad=False,
                    suffix_tok_id=self.suffix_tok_id,
                    prefix_tok_id=self.prefix_tok_id,
                    middle_tok_id=self.middle_tok_id,
                    pad_tok_id=self.pad_tok_id,
                )
                new_samples.append(permuted)

                sample = numpy.concatenate(new_samples)
            else:
                sample, self.np_rng = permute(
                    sample,
                    self.np_rng,
                    self.fim_rate,
                    self.fim_spm_rate,
                    self.tokenizer,
                    truncate_or_pad=False,
                    suffix_tok_id=self.suffix_tok_id,
                    prefix_tok_id=self.prefix_tok_id,
                    middle_tok_id=self.middle_tok_id,
                    pad_tok_id=self.pad_tok_id,
                )

            # Truncate or pad sequence to max-length
            diff = sample.shape[0] - sample_len
            if diff > 0:  # too long
                sample = sample[:sample_len]
            elif diff < 0:  # too short
                sample = numpy.concatenate([sample, numpy.full((-1 * diff), self.pad_tok_id)])

            assert sample.shape[0] == sample_len

        return sample, numpy.array(document_ids, dtype=numpy.int64)

    def _build_document_sample_shuffle_indices(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the
            shuffle index

        TODO: Explain the 80% threshold
        """

        path_to_cache = getattr(self.config, "path_to_cache")
        if path_to_cache is None:
            path_to_cache = os.path.join(self.indexed_dataset.path_prefix, "cache", f"{type(self).__name__}_indices")

        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )

        num_tokens_per_epoch = _get_num_tokens_per_epoch(self.indexed_dataset, self.indexed_indices)

        sequence_length = getattr(self.config, "sequence_length")

        num_epochs = _get_num_epochs(num_tokens_per_epoch, sequence_length, self.num_samples)

        log_rank_0(logging.INFO, f"> Tokens per epoch: {num_tokens_per_epoch}")
        log_rank_0(logging.INFO, f"> Number of epochs: {num_epochs}")

        if not cache_hit and self.caching_allowed:
            log_rank_0(logging.INFO, f"Build and save the {type(self).__name__} {self.index_split.name} indices")

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = ((num_epochs - 1) * num_tokens_per_epoch - 1) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (num_tokens_per_epoch - 1) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(threshold * num_samples_per_epoch)

                log_rank_0(logging.DEBUG, f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}")
                log_rank_0(logging.DEBUG, f"> threshold: {threshold}")
                log_rank_0(logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}")

            log_rank_0(logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}")

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the document index
            log_rank_0(
                logging.INFO, f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}"
            )
            t_beg = time.time()
            document_index = _build_document_index(
                self.indexed_indices, num_epochs, numpy_random_state, separate_final_epoch
            )
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the sample index
            log_rank_0(logging.INFO, f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}")
            t_beg = time.time()

            assert self.indexed_dataset.sequence_lengths.dtype == numpy.int32
            sample_index = build_sample_idx(
                self.indexed_dataset.sequence_lengths,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
            )
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the shuffle index
            log_rank_0(
                logging.INFO, f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}"
            )
            t_beg = time.time()
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices")

        log_rank_0(logging.INFO, f"\tLoad the document index from {os.path.basename(path_to_document_index)}")
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}")
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}")
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}")
        log_rank_0(logging.INFO, f"> total number of epochs: {num_epochs}")

        return document_index, sample_index, shuffle_index


def _get_num_tokens_per_epoch(indexed_dataset: MMapIndexedDataset, indices: numpy.ndarray) -> int:
    """Calculate the number of tokens in a single epoch

    Args:
        indexed_dataset (MMapIndexedDataset): The underlying MMapIndexedDataset

        indices (numpy.ndarray): The subset of indices into the underlying MMapIndexedDataset

    Returns:
        int: The number of tokens in a single epoch
    """
    return numpy.sum(indexed_dataset.sequence_lengths[indices])


def _get_num_epochs(num_tokens_per_epoch: int, seq_length: int, num_samples: int) -> int:
    """Calculate the number of epochs

    Args:
        num_tokens_per_epoch (int): The number of tokens in a single epoch

        seq_length (int): The sequence length in tokens

        num_samples (int): The total number of samples

    Returns:
        int: The number of epochs
    """
    num_epochs = 0
    num_tokens = 0
    while True:
        num_epochs += 1
        num_tokens += num_tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((num_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index

    TODO: Explain separate_final_epoch
    """

    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(documents.dtype)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines

        the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index

    TODO: Explain [0, num_samples) [num_samples, total_size) split
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(
    sample: numpy.ndarray,
    np_rng: numpy.random.RandomState,
    fim_rate: float,
    fim_spm_rate: float,
    tokenizer: AutoTokenizer,
    truncate_or_pad: bool = True,
    suffix_tok_id: int | None = None,
    prefix_tok_id: int | None = None,
    middle_tok_id: int | None = None,
    pad_tok_id: int | None = None,
):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    if np_rng.binomial(1, fim_rate):  # sample bernoulli dist
        contents = tokenizer.decode(sample)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = numpy.array([*tokenizer.encode(prefix)], dtype=numpy.int64)
        middle = numpy.array([*tokenizer.encode(middle)], dtype=numpy.int64)
        suffix = numpy.array([*tokenizer.encode(suffix)], dtype=numpy.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if (
                    suffix.shape[0] <= diff
                ):  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = numpy.concatenate([suffix, numpy.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = numpy.concatenate([[prefix_tok_id, suffix_tok_id], suffix, [middle_tok_id], prefix, middle])
        else:
            # PSM
            new_sample = numpy.concatenate([[prefix_tok_id], prefix, [suffix_tok_id], suffix, [middle_tok_id], middle])
    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample, np_rng
