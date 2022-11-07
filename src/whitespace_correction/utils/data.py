import json
import os
import pprint
import random
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import lmdb

import msgpack

import torch
from torch import distributed as dist
from torch.nn.utils import rnn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset
)

from whitespace_correction.model import tokenizer as toklib
from whitespace_correction.utils import common, constants, nlp, whitespace_correction
from whitespace_correction.utils.config import TrainConfig, ValConfig, TokenizerConfig
from whitespace_correction.utils.distributed import DistributedInfo

Sample = Dict[str, Any]
PreprocessingInputOutput = Union[Dict[str, Any], List[Dict[str, Any]]]
PreprocessingFn = Callable[[PreprocessingInputOutput], PreprocessingInputOutput]
TransformFn = Callable[[Dict[str, Any]], Dict[str, Any]]

logger = common.get_logger("DATA")


class Collate:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class PadCollate(Collate):
    def __init__(self, pad_values: Dict[str, int]):
        self.pad_values = pad_values

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        new_batch = defaultdict(list)
        for item in batch:
            for k, v in item.items():
                new_batch[k].append(v)
        return {
            k: rnn.pad_sequence(
                sequences=v,
                batch_first=True,
                padding_value=self.pad_values[k]
            ) if k in self.pad_values else v
            for k, v in new_batch.items()
        }


class DatasetMixin:
    def get_collate_fn(self) -> Collate:
        raise NotImplementedError()


class SequenceDatasetMixin(DatasetMixin):
    def __init__(self, pad_values: Dict[str, int]):
        self.pad_values = pad_values

    def get_lengths(self) -> List[int]:
        raise NotImplementedError()

    def get_collate_fn(self) -> Collate:
        return PadCollate(self.pad_values)


def _open_lmdb(lmdb_path: str) -> lmdb.Environment:
    env = lmdb.open(
        lmdb_path,
        map_size=int(10e11),  # approx. 100 GB
        subdir=False,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )
    return env


class PretokenizedDataset(Dataset, SequenceDatasetMixin):
    def __init__(
            self,
            lmdb_path: str,
            input_pad_token_id: int,
            target_pad_token_id: int,
            min_seq_length: int,
            max_seq_length: int,
            in_memory: bool = False,
            transform_fn: Optional[TransformFn] = None
    ):
        global logger
        self.lmdb_path = lmdb_path
        self.input_pad_token_id = input_pad_token_id
        self.target_pad_token_id = target_pad_token_id
        self.in_memory = in_memory
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.transform_fn = transform_fn

        assert self.min_seq_length <= self.max_seq_length, f"min sequence length must be smaller or equal to max " \
                                                           f"sequence length, but got {self.min_seq_length} " \
                                                           f"and {self.max_seq_length}"

        pad_values = {
            "input_ids": self.input_pad_token_id,
            "target_input_ids": self.target_pad_token_id,
            "labels": -1
        }
        super().__init__(pad_values=pad_values)

        env = _open_lmdb(lmdb_path=lmdb_path)

        self.txn = env.begin(write=False)

        _length = msgpack.loads(self.txn.get(b"__len__"))
        _lmdb_lengths_keys = msgpack.loads(self.txn.get(b"__lengths__"))
        _lmdb_lengths = []
        for _length_key in _lmdb_lengths_keys:
            _lmdb_lengths.extend(msgpack.loads(self.txn.get(_length_key)))

        _lmdb_keys_keys = msgpack.loads(self.txn.get(b"__keys__"))
        self._lmdb_keys = []
        for _key_key in _lmdb_keys_keys:
            self._lmdb_keys.extend(msgpack.loads(self.txn.get(_key_key)))

        assert len(self._lmdb_keys) == len(_lmdb_lengths) == _length, \
            "found different number of lengths and keys in lmdb"

        self._lengths = []
        self._indices = []
        for idx in range(len(_lmdb_lengths)):
            length = _lmdb_lengths[idx]
            if length < self.min_seq_length or length > self.max_seq_length:
                continue

            self._lengths.append(length)
            self._indices.append(idx)

        self.lmdb_files = msgpack.loads(self.txn.get(b"__files__"))

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            logger.info(f"The following files (first 20) are stored in the lmdb database '{lmdb_path}':\n"
                        f"{pprint.pformat(sorted(self.lmdb_files[:20]))}")

            overflowing = _length - len(self._indices)
            logger.info(f"Found {overflowing} out of {_length} sequences in the data that are "
                        f"> {self.max_seq_length} or < {self.min_seq_length}. Skipping them.")

        self.max_samples = int(os.environ.get("MAX_DATA_SAMPLES", len(self._indices)))

        self._data = None
        if self.in_memory:
            self._data = []
            start = time.monotonic()
            length = len(self)
            for idx in range(length):
                packed = self.txn.get(self._lmdb_keys[self._indices[idx]])
                data = msgpack.loads(packed)
                self._data.append(data)

                if (idx + 1) % (length / 100) == 0:
                    end = time.monotonic()
                    logger.info(f"Loaded {(idx + 1) * 100 / length:.2f}% of the data into memory. "
                                f"{common.eta_minutes(end - start / 60, idx + 1, length)}")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__
        state["txn"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        env = _open_lmdb(lmdb_path=self.lmdb_path)
        self.txn = env.begin(write=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.txn is None:
            env = _open_lmdb(lmdb_path=self.lmdb_path)
            self.txn = env.begin(write=False)

        if self.in_memory:
            data = self._data[idx]
        else:
            packed = self.txn.get(self._lmdb_keys[self._indices[idx]])
            data = msgpack.loads(packed)

        if self.transform_fn is not None:
            data = self.transform_fn(data)

        return {k: torch.tensor(v) if k in self.pad_values else v for k, v in data.items()}

    def __len__(self) -> int:
        return min(len(self._indices), self.max_samples)

    def get_lengths(self) -> List[int]:
        return self._lengths


class MaxSeqLenDataset(Dataset, SequenceDatasetMixin):
    def __init__(self,
                 max_seq_len: int,
                 max_sequences: int,
                 input_vocab_size: int,
                 output_vocab_size: int):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_seq_len = max_seq_len
        self.max_sequences = max_sequences
        self.rand = torch.Generator()
        self.rand.manual_seed(dist.get_rank() if dist.is_initialized() else 22)
        pad_values = {"input_ids": 0,
                      "target_input_ids": 0,
                      "labels": -1}
        super().__init__(pad_values=pad_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.randint(high=self.input_vocab_size,
                                           size=(self.max_seq_len // (idx % 4 + 1),),
                                           generator=self.rand),
                "target_input_ids": torch.randint(high=self.output_vocab_size,
                                                  size=(self.max_seq_len // (idx % 4 + 1),),
                                                  generator=self.rand)}

    def __len__(self) -> int:
        return self.max_sequences

    def get_lengths(self) -> List[int]:
        return [self.max_seq_len // (i % 4 + 1) for i in range(len(self))]


class BucketSampler(Sampler):
    def __init__(self,
                 data_source: Union[PretokenizedDataset, Subset],
                 max_tokens: int,
                 min_seq_len: int,
                 max_seq_len: int,
                 bucket_span: int,
                 shuffle: bool = False,
                 seed: int = 22) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.bucket_span = bucket_span
        self.seed = seed

        assert self.min_seq_len <= self.max_seq_len, f"min sequence length must be smaller or equal to max sequence " \
                                                     f"length, but got {self.min_seq_len} and {self.max_seq_len}"
        assert self.max_seq_len <= self.max_tokens, "max sequence length cannot be greater " \
                                                    f"than max number of tokens per batch, but got {self.max_seq_len}" \
                                                    f"and {self.max_tokens}"
        assert self.max_seq_len % self.bucket_span == 0, "max sequence length must be divisible by bucket span"

        self.rand = random.Random(self.seed)

        # initialize batches to empty list and epoch to -1
        # call set epoch which will build the batches for epoch 0
        self.batches = []
        self.epoch = -1
        self.set_epoch(0)

    def _build_batches(self) -> List[List[int]]:
        self.rand.seed(self.seed + self.epoch)

        if isinstance(self.data_source, PretokenizedDataset):
            indices_lengths = list(zip(list(range(len(self.data_source))), self.data_source.get_lengths()))

        elif isinstance(self.data_source, Subset):
            assert isinstance(self.data_source.dataset,
                              SequenceDatasetMixin), f"data source is of type Subset, but the" \
                                                     f"underlying dataset does not inherit from " \
                                                     f"SequenceDatasetMixin, got type {type(self.data_source.dataset)}"

            dataset_lengths = self.data_source.dataset.get_lengths()
            subset_indices = self.data_source.indices
            subset_lengths = [dataset_lengths[idx] for idx in subset_indices]
            indices_lengths = list(zip(list(range(len(self.data_source))), subset_lengths))

        else:
            raise ValueError(f"BucketSampler needs a PretokenizedDataset or a "
                             f"Subset of a PretokenizedDataset as data source, but got {type(self.data_source)}")

        num_buckets = (self.max_seq_len - self.min_seq_len) // self.bucket_span + 1
        bucket_max_lengths = [
            min((bucket_idx + 1) * self.bucket_span - 1 + self.min_seq_len, self.max_seq_len)
            for bucket_idx in range(num_buckets)
        ]
        bucket_max_batch_samples = [
            self.max_tokens // max(1, bucket_max_lengths[bucket_idx])
            for bucket_idx in range(num_buckets)
        ]
        batch_buckets: List[List[List[int]]] = [[] for _ in range(num_buckets)]

        if self.shuffle:
            self.rand.shuffle(indices_lengths)

        for idx, length in indices_lengths:
            bucket_idx = (length - self.min_seq_len) // self.bucket_span

            if len(batch_buckets[bucket_idx]) == 0:
                batch_buckets[bucket_idx].append([])

            if len(batch_buckets[bucket_idx][-1]) < bucket_max_batch_samples[bucket_idx]:
                batch_buckets[bucket_idx][-1].append(idx)
            else:
                batch_buckets[bucket_idx].append([idx])

        # unfold batches into list of lists
        batches = [batch for bucket in batch_buckets for batch in bucket if len(batch) > 0]
        if self.shuffle:
            self.rand.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator:
        for batch in self.batches:
            yield batch

    def set_epoch(self, e: int) -> None:
        if e == self.epoch:
            return
        self.epoch = e
        self.batches = self._build_batches()

    def __len__(self) -> int:
        return len(self.batches)


# modified version of
# https://catalyst-team.github.io/catalyst/_modules/catalyst/data/dataset/torch.html#DatasetFromSampler
class SamplerDataset(Dataset):
    def __init__(self, sampler: Sampler) -> None:
        super().__init__()
        self.sampler = sampler
        self.sampler_indices = None

    def __getitem__(self, idx: int) -> Any:
        if self.sampler_indices is None:
            self.sampler_indices = list(self.sampler)
        return self.sampler_indices[idx]

    def __len__(self) -> int:
        return len(self.sampler)


# modified version of
# https://catalyst-team.github.io/catalyst/_modules/catalyst/data/sampler.html#DistributedSamplerWrapper
class DistributedDynamicSampler(DistributedSampler):
    def __init__(self,
                 sampler: Sampler,
                 seed: int,
                 drop_last: bool = False,
                 shuffle: bool = True) -> None:
        super().__init__(SamplerDataset(sampler),
                         shuffle=shuffle,
                         seed=seed,
                         drop_last=drop_last)
        self.sampler = sampler
        self.steps_to_fast_forward = 0

    def __iter__(self) -> List[int]:
        self.dataset = SamplerDataset(self.sampler)

        dist_indices = list(super().__iter__())
        sampler_indices = self.dataset

        for idx in dist_indices[self.steps_to_fast_forward:]:
            yield sampler_indices[idx]

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def set_steps_to_fast_forward(self, steps: int) -> None:
        self.steps_to_fast_forward = steps

    def __len__(self) -> int:
        return super().__len__() - self.steps_to_fast_forward


def get_data_from_config(
        train_config: TrainConfig,
        val_config: ValConfig,
        info: DistributedInfo,
        seed: int,
        input_pad_token_id: int,
        target_pad_token_id: int
) -> Tuple[DataLoader, DataLoader]:
    transform_fn = get_transform_fn("swap_inputs_and_targets") if train_config.swap_inputs_and_targets else None

    dataset = PretokenizedDataset(
        lmdb_path=train_config.train_data,
        input_pad_token_id=input_pad_token_id,
        target_pad_token_id=target_pad_token_id,
        min_seq_length=train_config.min_seq_length,
        max_seq_length=train_config.max_seq_length,
        in_memory=train_config.in_memory,
        transform_fn=transform_fn
    )

    if isinstance(val_config.val_data, str):
        assert len(val_config.val_data) > 0, "val data cannot be empty"
        val_dataset = PretokenizedDataset(
            lmdb_path=val_config.val_data,
            input_pad_token_id=input_pad_token_id,
            target_pad_token_id=target_pad_token_id,
            min_seq_length=train_config.min_seq_length,
            max_seq_length=train_config.max_seq_length,
            in_memory=train_config.in_memory,
            transform_fn=transform_fn
        )
        train_dataset = dataset
        train_collate = train_dataset.get_collate_fn()
        val_collate = val_dataset.get_collate_fn()

    elif isinstance(val_config.val_data, float) or isinstance(val_config.val_data, int):
        rand = random.Random(seed)
        indices = list(range(len(dataset)))
        rand.shuffle(indices)
        if isinstance(val_config.val_data, float):
            assert 0 < val_config.val_data < 1, "val data has to be a float between 0 and 1"
            val_upper = int(len(indices) * val_config.val_data)
        else:
            assert val_config.val_data > 0, "val data has to be an int larger 0"
            val_upper = val_config.val_data

        val_indices = indices[:val_upper]
        train_indices = indices[val_upper:]

        val_dataset = Subset(dataset, indices=val_indices)
        train_dataset = Subset(dataset, indices=train_indices)

        train_collate = val_collate = dataset.get_collate_fn()

    else:
        raise ValueError(f"invalid type {type(val_config.val_data)} for val_data, "
                         f"only str, float and int are supported")

    if train_config.batch_max_tokens:
        assert train_config.batch_size is None, "cannot specify batch_max_tokens and batch_size together"
        train_sampler = BucketSampler(
            data_source=train_dataset,
            max_tokens=train_config.batch_max_tokens // info.world_size,
            min_seq_len=train_config.min_seq_length,
            max_seq_len=train_config.max_seq_length,
            bucket_span=4,
            shuffle=True,
            seed=seed
        )
        val_sampler = BucketSampler(
            data_source=val_dataset,
            max_tokens=train_config.batch_max_tokens // info.world_size,
            min_seq_len=train_config.min_seq_length,
            max_seq_len=train_config.max_seq_length,
            bucket_span=4,
            shuffle=False,
            seed=seed
        )

    else:
        assert train_config.batch_size is not None, "one of batch_max_tokens or batch_size must be specified"
        train_generator = torch.Generator()
        train_generator = train_generator.manual_seed(seed)
        train_sampler = BatchSampler(
            sampler=RandomSampler(train_dataset, generator=train_generator),
            batch_size=max(train_config.batch_size // info.world_size, 1),
            drop_last=True
        )
        val_sampler = BatchSampler(
            sampler=SequentialSampler(val_dataset),
            batch_size=max(train_config.batch_size // info.world_size, 1),
            drop_last=False
        )

    train_sampler_dist = DistributedDynamicSampler(sampler=train_sampler, shuffle=True, seed=seed, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler_dist,
        pin_memory=True,
        num_workers=train_config.num_workers,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        pin_memory=True,
        num_workers=0,
        collate_fn=val_collate
    )

    return train_loader, val_loader


_TransformMethods = {"swap_inputs_and_targets"}


def get_transform_fn(transform_method: str) -> TransformFn:
    if transform_method == "swap_inputs_and_targets":
        def _swap(item: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            assert "input_ids" in item and "target_input_ids" in item
            item["input_ids"], item["target_input_ids"] = item["target_input_ids"], item["input_ids"]
            return item

        return _swap
    else:
        raise NotImplementedError(
            f"Transform method {transform_method} not implemented. Use one of {_TransformMethods}.")


def chain_preprocessing_fns(preprocessing_fns: List[PreprocessingFn]) -> PreprocessingFn:
    """

    Allows to chain multiple corruption functions.

    :param preprocessing_fns: List of corruption functions
    :return: new corruption function that chains the given functions in order
    """

    def _chained_preprocessing_fn(item: PreprocessingInputOutput) -> PreprocessingInputOutput:
        for p_fn in preprocessing_fns:
            item = p_fn(item)
        return item

    return _chained_preprocessing_fn


def get_preprocessing_fn(preprocessing_method: str, **kwargs: Any) -> PreprocessingFn:
    """

    Assemble the specified corruption function using the specified keyword arguments.

    :param preprocessing_method: name of the corruption method
    :param kwargs: additional arguments which get passed to the specified corruption method function
    :return: corruption function
    """
    if preprocessing_method == "replace_corruption":
        return _replace_corruption(**kwargs)

    elif preprocessing_method == "replace_words_corruption":
        return _replace_words_corruption(**kwargs)

    elif preprocessing_method == "merge_split_swap_insert_delete_words_corruption":
        return _merge_split_swap_insert_delete_words_corruption(**kwargs)

    elif preprocessing_method == "edit_tokens_corruption":
        return _edit_tokens_corruption(**kwargs)

    elif preprocessing_method == "whitespace_corruption":
        return _whitespace_corruption(**kwargs)

    elif preprocessing_method == "whitespace_correction":
        return _whitespace_correction(**kwargs)

    elif preprocessing_method == "whitespace_subsampling":
        return _whitespace_subsampling(**kwargs)

    elif preprocessing_method == "character_masked_language_modeling":
        return _character_masked_language_modeling(**kwargs)

    elif preprocessing_method == "join_sequences":
        return _join_sequences(**kwargs)

    elif preprocessing_method == "switch":
        return _switch(**kwargs)

    elif preprocessing_method == "skip":
        return _skip()

    elif preprocessing_method == "drop":
        return _drop(**kwargs)

    else:
        raise NotImplementedError(f"Preprocessing method {preprocessing_method} not implemented")


def _whitespace_subsampling(
        length: int,
        unit: str = "char",
        seed: int = 22
) -> PreprocessingFn:
    assert unit in {"char", "byte"}, f"unknown unit {unit}, must be one of char or byte"
    rand = random.Random(seed)

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _sub(item: Dict[str, Any]) -> Dict[str, Any]:
            assert "sequence" in item and ("labels" in item or "target_sequence" in item)

            item["sequence"] = whitespace_correction.clean_sequence(item["sequence"])

            if unit == "char":
                start, end = whitespace_correction.random_character_substring(item["sequence"], length, rand)
            else:
                start, end = whitespace_correction.random_byte_substring(item["sequence"], length, rand)

            item["sequence"] = item["sequence"][start:end]
            if "labels" in item:
                item["labels"] = [0] + item["labels"][start + 1: end + 1] + [0]
            else:
                target_sequence = whitespace_correction.clean_sequence(item["target_sequence"])
                target_start, target_end = whitespace_correction.find_substring_ignoring_spaces(
                    target_sequence, item["sequence"]
                )
                item["target_sequence"] = target_sequence[target_start: target_end]
                assert item["sequence"].replace(" ", "") == item["target_sequence"].replace(" ", "")

            return item

        if isinstance(seq, list):
            seq = [_sub(t.copy()) for t in seq]
        else:
            seq = _sub(seq.copy())
        return seq

    return _preprocessing_fn


def _drop(keys: List[str]) -> PreprocessingFn:
    """

    Removes the specified keys from the input dictionary

    :param keys: name of the keys to remove
    :return:
    """

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _drop_function(item: Dict[str, str]) -> Dict[str, str]:

            for key in keys:
                item.pop(key, None)

            return item

        if isinstance(seq, list):
            seq = [_drop_function(t.copy()) for t in seq]
        else:
            seq = _drop_function(seq.copy())
        return seq

    return _preprocessing_fn


def _skip() -> PreprocessingFn:
    """

    Do nothing (useful to use together with switch to randomly apply some preprocessing)

    :return:
    """

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        return seq

    return _preprocessing_fn


def _switch(functions: List[Dict] = None,
            prob: List[float] = None,
            seed: int = 22) -> PreprocessingFn:
    """

    Preprocessing function that chooses one of the given preprocessing functions with the corresponding probability.

    :param functions: list of dictionaries that define other preprocessing functions
    :param prob: list of probabilities with which we choose each preprocessing function (must sum to 1)
    :param seed: random seed
    """
    assert functions is not None and prob is not None, "functions and prob cannot be None"
    assert len(functions) == len(prob), "specify a probability for each switched preprocessing function"
    assert sum(prob) == 1, "probabilities must sum to 1"

    rand = random.Random(seed)

    fns = []
    for fn_def in functions:
        assert "type" in fn_def, "type must be specified for each switched preprocessing function"

        kwargs = fn_def.get("arguments", {})

        fn = get_preprocessing_fn(fn_def["type"], **kwargs)
        fns.append(fn)

    indices = list(range(len(fns)))

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _switch_functions(item: Dict[str, str]) -> Dict[str, str]:

            idx = rand.choices(indices, weights=prob, k=1)[0]

            return fns[idx](item)

        if isinstance(seq, list):
            seq = [_switch_functions(t.copy()) for t in seq]
        else:
            seq = _switch_functions(seq.copy())
        return seq

    return _preprocessing_fn


def _character_masked_language_modeling(
        tokenizer: Optional[Dict[str, Any]] = None,
        word_p: float = 0.15,
        full_word_p: float = 0.5,
        seed: int = 22
) -> PreprocessingFn:
    """

    Preprocessing method for masked language modeling.

    :param word_p: probability of masking a word (split by whitespace)
    :param full_word_p: probability of masking a word completely
            (otherwise uniform(0, len(word) - 1) chars of the word will be masked)
    :param seed: random seed
    """
    tok = toklib.get_tokenizer_from_config(TokenizerConfig.from_dict(tokenizer))
    unk_token_id = tok.token_to_id(constants.UNK)
    rand = random.Random(seed)

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _mlm(item: Dict[str, Any]) -> Dict[str, Any]:
            assert "sequence" in item
            sequence = whitespace_correction.clean_sequence(item["sequence"])

            words = sequence.split()
            labels = [-1]
            masked_words = []
            for i, word in enumerate(words):
                if rand.random() < word_p:
                    if rand.random() < full_word_p:
                        # mask the full word
                        masked_words.append(constants.MASK * len(word))
                        labels.extend([tok.token_to_id(char) or unk_token_id for char in word])
                    else:
                        masked_chars = list(word)
                        char_indices = list(range(len(masked_chars)))
                        rand.shuffle(char_indices)
                        num_masked_chars = rand.randint(1, max(1, len(masked_chars) - 1))
                        mask_indices = char_indices[:num_masked_chars]
                        masked_labels = [-1] * len(masked_chars)
                        for idx in mask_indices:
                            masked_labels[idx] = tok.token_to_id(masked_chars[idx]) or unk_token_id
                            masked_chars[idx] = constants.MASK
                        masked_words.append("".join(masked_chars))
                        labels.extend(masked_labels)
                else:
                    masked_words.append(word)
                    labels.extend([-1] * len(word))

                if i < len(words) - 1:
                    labels.append(-1)  # for the whitespace after each word

            labels.append(-1)
            sequence = " ".join(masked_words)

            item["sequence"] = sequence
            item["labels"] = labels
            item.pop("target_sequence", None)

            return item

        if isinstance(seq, list):
            seq = [_mlm(t.copy()) for t in seq]
        else:
            seq = _mlm(seq.copy())
        return seq

    return _preprocessing_fn


def _join_sequences(target_first: bool = False) -> PreprocessingFn:
    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _join(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" and "target_sequence" in item

            sequence = item["sequence"]
            target_sequence = item["target_sequence"]

            if target_first:
                sequence, target_sequence = target_sequence, sequence

            item["sequence"] = sequence + f" {constants.SEP} " + target_sequence
            item.pop("target_sequence", None)

            return item

        if isinstance(seq, list):
            seq = [_join(t.copy()) for t in seq]
        else:
            seq = _join(seq.copy())
        return seq

    return _preprocessing_fn


def _edit_distance(tokenizer: str = "") -> PreprocessingFn:
    _edit_ops = _edit_operations(tokenizer=tokenizer)

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _ed(item: Dict[str, Any]) -> Dict[str, Any]:

            edit_ops = _edit_ops(item)
            assert isinstance(edit_ops, dict)
            edit_ops_labels: List[int] = edit_ops["labels"]
            edit_distance = sum(edit_ops_labels)

            item["labels"] = edit_distance
            item.pop("target_sequence", None)

            return item

        if isinstance(seq, list):
            seq = [_ed(t.copy()) for t in seq]
        else:
            seq = _ed(seq.copy())
        return seq

    return _corruption_fn


def _edit_operations(tokenizer: str = "") -> PreprocessingFn:
    """

    :param tokenizer: tokenizer to use
    :return: corruption function with signature CORRUPTION_FN
    """
    tok = toklib.get_tokenizer_from_config(tokenizer)

    import difflib

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _get_edit_operations(item: Dict[str, Any]) -> Dict[str, Any]:
            assert "sequence" in item and "target_sequence" in item

            encodings = tok.encode_batch([item["sequence"], item["target_sequence"]])
            seq_ids, tgt_ids = encodings[0].ids, encodings[1].ids

            labels = []

            sm = difflib.SequenceMatcher(a=seq_ids, b=tgt_ids)
            matches = sm.get_matching_blocks()
            labels.extend([0] * matches[0].size)

            for i in range(1, len(matches)):
                match = matches[i]
                last_match = matches[i - 1]
                labels.extend([1] * (match.a - (last_match.a + last_match.size)))
                labels.extend([0] * match.size)

            assert len(labels) == len(seq_ids)

            item["labels"] = labels
            item.pop("target_sequence", None)

            return item

        if isinstance(seq, list):
            seq = [_get_edit_operations(t.copy()) for t in seq]
        else:
            seq = _get_edit_operations(seq.copy())
        return seq

    return _corruption_fn


def _edit_tokens_corruption(p: float = 0.1,
                            seed: int = 22) -> PreprocessingFn:
    """

    Corrupt by applying edits (delete, replace, swap , insert) to tokens (split by whitespace).

    :param p: probability that a token is corrupted
    :param seed: fix the random seed
    :return: corruption function with signature CORRUPTION_FN
    """
    rand = random.Random(seed)

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _corrupt_token(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])

            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            tokens = [nlp.edit_token(token, rand)[0] if rand.random() < p else token
                      for token in sequence.split(" ")]

            item["sequence"] = " ".join(tokens)

            return item

        if isinstance(seq, list):
            seq = [_corrupt_token(t.copy()) for t in seq]
        else:
            seq = _corrupt_token(seq.copy())
        return seq

    return _corruption_fn


def _whitespace_correction(
        output_type: str = "label"
) -> PreprocessingFn:
    """

    Adds whitespace correction labels to the data items.

    :param output_type: one of {label, char}
    :return: corruption function with signature CORRUPTION_FN
    """
    assert output_type in {"label", "char"}

    def _preprocessing_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _ws_cor(item: Dict[str, Any]) -> Dict[str, Any]:
            assert "sequence" in item and "target_sequence" in item

            sequence = whitespace_correction.clean_sequence(item["sequence"])
            target_sequence = whitespace_correction.clean_sequence(item["target_sequence"])

            item["sequence"] = sequence

            if output_type == "label":
                labels = whitespace_correction.get_whitespace_operations(sequence, target_sequence)
                item["labels"] = [0] + labels + [0]
                item.pop("target_sequence", None)
            else:
                item["target_sequence"] = target_sequence
                item.pop("labels", None)
                assert item["sequence"].replace(" ", "") == item["target_sequence"].replace(" ", "")

            return item

        if isinstance(seq, list):
            seq = [_ws_cor(t.copy()) for t in seq]
        else:
            seq = _ws_cor(seq.copy())
        return seq

    return _preprocessing_fn


def _whitespace_corruption(iw_p: float = 0.2,
                           dw_p: float = 0.5,
                           no_ws: bool = False,
                           full_ws: bool = False,
                           seed: int = 22) -> PreprocessingFn:
    """

    Corruption by randomly inserting whitespaces into
    or randomly deleting whitespaces from sequences

    :param iw_p: probability that a whitespace is inserted at each non-whitespace position
    :param dw_p: probability that a whitespace is deleted
    :param no_ws: if True, all whitespaces will be removed from sequence
    :param full_ws: if True, whitespaces will be inserted after every character in the sequence
    :param seed: fix the random seed
    :return: corruption function with signature CORRUPTION_FN
    """
    rand = random.Random(seed)
    assert not (no_ws and full_ws), "no_ws and full_ws cannot both be true"

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _no_ws(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])
            item["sequence"] = re.sub(r"\s", "", sequence)

            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            return item

        def _full_ws(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])
            item["sequence"] = " ".join(re.sub(r"\s", "", sequence))

            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            return item

        def _insert_or_delete(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])

            new_s = ""
            sequence_ptr = 0
            while sequence_ptr < len(sequence):
                char = sequence[sequence_ptr]
                prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else ""
                r = rand.random()

                if char == " ":
                    if r < dw_p:
                        pass
                    else:
                        new_s += char
                elif prev_char != " " and r < iw_p:
                    new_s += " " + char
                else:
                    new_s += char

                sequence_ptr += 1

            item["sequence"] = new_s

            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            return item

        as_list = isinstance(seq, list)
        new_seq = []

        if not as_list:
            seq = [seq]

        for s in seq:
            if no_ws:
                s = _no_ws(s.copy())
            elif full_ws:
                s = _full_ws(s.copy())
            else:
                s = _insert_or_delete(s.copy())
            new_seq.append(s)

        if not as_list:
            return new_seq[0]
        else:
            return new_seq

    return _corruption_fn


def _replace_words_corruption(replace: Union[str, Dict[str, List[str]]] = None,
                              p: float = 0.1,
                              seed: int = 22) -> PreprocessingFn:
    """

    Corruption by replacing words with a random choice of provided replacement words.
    Note that only words that appear in the replace dictionary will be replaced.

    :param replace: mapping of words to possible replacement words
    :param p: probability that a word is replaced
    :return: corruption function with signature CORRUPTION_FN
    """
    rand = random.Random(seed)

    pattern = r'^[!"#$%&\'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~]*|[!"#$%&\'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~]*$'

    if isinstance(replace, str):
        with open(replace, "r", encoding="utf8") as f:
            replace_dict = json.load(f)
    elif isinstance(replace, dict):
        replace_dict = replace
    else:
        raise ValueError("replace must be either a path to a file or a dictionary of replacements")

    logger.info(f"[REPLACE_WORDS_CORRUPTION] Found {len(replace_dict)} words that can be replaced with a total of "
                f"{sum(len(v) for v in replace_dict.values())} words")

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _rep(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])

            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            words = sequence.split(" ")
            for i in range(len(words)):
                org_word = words[i]
                word = re.sub(pattern, "", org_word)

                if rand.random() < p:
                    if word in replace_dict:
                        replacement_word = rand.choice(replace_dict[word])
                    elif word.lower() in replace_dict:
                        replacement_word = rand.choice(replace_dict[word.lower()])
                    else:
                        continue

                    org_word = org_word.replace(word, replacement_word)

                words[i] = org_word

            item["sequence"] = " ".join(words)

            return item

        if isinstance(seq, list):
            seq = [_rep(t.copy()) for t in seq]
        else:
            seq = _rep(seq.copy())
        return seq

    return _corruption_fn


def _merge_split_swap_insert_delete_words_corruption(words: Union[str, List[str]] = None,
                                                     p: float = 0.1,
                                                     seed: int = 22) -> PreprocessingFn:
    rand = random.Random(seed)

    choices = ["merge", "split", "swap", "insert", "delete"]

    if isinstance(words, str):
        with open(words, "r", encoding="utf8") as f:
            words_list = f.readlines()
    elif isinstance(words, list):
        words_list = words
    else:
        raise ValueError("words must be either a path to a file or a list of words")

    words_list = [word.strip() for word in words_list]

    logger.info(f"[MERGE_SPLIT_SWAP_INSERT_DELETE_WORDS_CORRUPTION] Found {len(words_list)} words "
                f"that can be inserted")

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _merge_split_swap_insert_delete(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            sequence = whitespace_correction.clean_sequence(item["sequence"])
            if "target_sequence" not in item:
                item["target_sequence"] = sequence

            words = sequence.split(" ")
            new_words = []

            word_pointer = 0
            while word_pointer < len(words):
                word = words[word_pointer]

                if rand.random() < p:
                    choice = rand.choices(choices, weights=[0.35, 0.35, 0.1, 0.1, 0.1])[0]

                    if choice == "merge":
                        # cant merge if we are at end of sentence
                        if word_pointer == len(words) - 1:
                            new_words.append(word)
                            word_pointer += 1
                        else:
                            next_word = words[word_pointer + 1]
                            new_words.append(word + next_word)
                            word_pointer += 2

                    elif choice == "split":
                        # cant split with 1 letter words
                        if len(word) == 1:
                            new_words.append(word)
                        else:
                            split_idx = rand.randint(1, len(word) - 1)
                            new_words.append(word[:split_idx])
                            new_words.append(word[split_idx:])
                        word_pointer += 1

                    elif choice == "swap":
                        # cant swap if we are at end of sentence
                        if word_pointer == len(words) - 1:
                            new_words.append(word)
                            word_pointer += 1
                        else:
                            next_word = words[word_pointer + 1]
                            new_words.append(next_word)
                            new_words.append(word)
                            word_pointer += 2

                    elif choice == "insert":
                        new_words.append(word)
                        random_word = random.choice(words_list)
                        new_words.append(random_word)
                        word_pointer += 1

                    else:  # delete
                        word_pointer += 1
                else:
                    new_words.append(word)
                    word_pointer += 1

            item["sequence"] = " ".join(new_words)
            return item

        if isinstance(seq, list):
            seq = [_merge_split_swap_insert_delete(t.copy()) for t in seq]
        else:
            seq = _merge_split_swap_insert_delete(seq.copy())
        return seq

    return _corruption_fn


def _replace_corruption(replace: Dict[str, str], p: float = 0.5, seed: int = 22) -> PreprocessingFn:
    """

    Corruption by replacing sequences with other sequences

    :param replace: mapping of strings that should be replaced with each other
    :param p: probability that a match is replaced
    :return: corruption function with signature CORRUPTION_FN
    """
    rand = random.Random(seed)

    def _corruption_fn(seq: PreprocessingInputOutput) -> PreprocessingInputOutput:
        def _rep(item: Dict[str, str]) -> Dict[str, str]:
            assert "sequence" in item, "need key sequence in dictionary to corrupt"
            item["target_sequence"] = item["sequence"]
            new_s = item["sequence"]
            for fr, to in replace.items():
                new_s_len = len(new_s)
                diff = 0
                for match in re.finditer(fr, new_s):
                    if rand.random() > p:
                        continue
                    new_s = new_s[:match.start() + diff] + to + new_s[match.end() + diff:]
                    diff = len(new_s) - new_s_len
            item["sequence"] = new_s
            return item

        if isinstance(seq, list):
            seq = [_rep(t.copy()) for t in seq]
        else:
            seq = _rep(seq.copy())
        return seq

    return _corruption_fn
