import itertools
import os
from collections import Counter
from typing import Any, Dict, List

import pytest

import torch

from trt.utils import constants, data
from trt.utils.data import get_preprocessing_fn
from trt.utils.tokenization_repair import TokenizationRepairTokens

BASE_DIR = os.path.dirname(__file__)


class TestData:
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_pretokenized_dataset(self, in_memory: bool) -> None:
        dummy_path = os.path.join(BASE_DIR, "..", "data", "lmdb", "dummy_lmdb")

        dataset = data.PretokenizedDataset(lmdb_path=dummy_path,
                                           pad_token_id=0,
                                           in_memory=in_memory,
                                           max_seq_length=512,
                                           min_seq_length=0)
        assert len(dataset) == 2

        for sample in dataset:
            assert not torch.equal(sample["input_ids"], sample["target_input_ids"])

        dataset = data.PretokenizedDataset(lmdb_path=dummy_path,
                                           pad_token_id=0,
                                           in_memory=in_memory,
                                           max_seq_length=512,
                                           min_seq_length=512)
        assert len(dataset) == 0

        with pytest.raises(AssertionError):
            _ = data.PretokenizedDataset(lmdb_path=dummy_path,
                                         pad_token_id=0,
                                         in_memory=in_memory,
                                         max_seq_length=256,
                                         min_seq_length=512)

    @pytest.mark.parametrize("in_memory", [True, False])
    def test_transform_fns(self, in_memory: bool) -> None:
        dummy_path = os.path.join(BASE_DIR, "..", "data", "lmdb", "dummy_lmdb")

        dataset = data.PretokenizedDataset(lmdb_path=dummy_path,
                                           pad_token_id=0,
                                           in_memory=in_memory,
                                           max_seq_length=512,
                                           min_seq_length=0)

        transform_fn = data.get_transform_fn("swap_inputs_and_targets")

        for sample in dataset:
            transformed_sample = transform_fn(sample)
            for k in sample.keys():
                assert torch.equal(sample[k], transformed_sample[k])

    @pytest.mark.parametrize("replace", list("aeisth "))
    @pytest.mark.parametrize("replace_with", list("aeisth "))
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_replace_corruption(self, replace: str, replace_with: str, seed: int) -> None:
        if replace == replace_with:
            replace_with = ""

        corruption_fn = get_preprocessing_fn("replace_corruption",
                                             replace={replace: replace_with},
                                             p=1.0,
                                             seed=seed)

        test_item = {"sequence": "This is a test"}

        out_item = corruption_fn(test_item)

        assert out_item["target_sequence"] == test_item["sequence"]
        assert replace not in out_item["sequence"]

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_replace_words_corruption(self, seed: int) -> None:
        replacements = ["apple",
                        "banana",
                        "orange",
                        "pineapple"]
        corruption_fn = get_preprocessing_fn("replace_words_corruption",
                                             replace={"test": replacements},
                                             p=1.0,
                                             seed=seed)

        test_item = {"sequence": "This is a test"}

        out_item = corruption_fn(test_item)

        assert out_item["target_sequence"] == test_item["sequence"]
        assert any(r in out_item["sequence"] for r in replacements)

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_merge_split_swap_insert_delete_words_corruption(self, seed: int) -> None:
        words = ["apple",
                 "banana",
                 "orange",
                 "pineapple"]
        corruption_fn = get_preprocessing_fn("merge_split_swap_insert_delete_words_corruption",
                                             words=words,
                                             p=1.0,
                                             seed=seed)

        test_item = {"sequence": "This is a test"}

        out_item = corruption_fn(test_item)

        assert out_item["target_sequence"] == test_item["sequence"]
        assert out_item["sequence"] != test_item["sequence"]

    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("no_ws", [True, False])
    def test_whitespace_corruption(self, seed: int, no_ws: bool) -> None:
        corruption_fn = get_preprocessing_fn("whitespace_corruption",
                                             iw_p=0.5,
                                             dw_p=0.5,
                                             no_ws=no_ws,
                                             seed=seed)

        test_item = {"sequence": "This is a test"}

        out_item = corruption_fn(test_item)

        assert out_item["target_sequence"] == test_item["sequence"]
        diff = Counter(test_item["sequence"]) - Counter(out_item["sequence"])
        assert len(diff) <= 1
        if len(diff) == 1:
            assert " " in diff

        if no_ws:
            assert " " not in out_item["sequence"]

    def test_skip(self) -> None:
        preprocessing_fn = get_preprocessing_fn("skip")

        test_item = {"sequence": "This is a test"}

        out_item = preprocessing_fn(test_item)

        assert set(out_item.keys()) == set(test_item.keys())
        assert all(out_item[k] == test_item[k] for k in out_item.keys())

    @pytest.mark.parametrize("drop_keys", [perm
                                           for i in range(4)
                                           for perm in
                                           itertools.permutations(
                                               ["sequence", "key1", "key2", "not_found"],
                                               r=i)
                                           ]
                             )
    def test_drop(self, drop_keys: List[str]) -> None:
        preprocessing_fn = get_preprocessing_fn("drop", keys=drop_keys)

        test_item = {"sequence": "This is a test",
                     "key1": "value1",
                     "key2": "value2"}

        out_item = preprocessing_fn(test_item)

        keys = set(test_item.keys()) - set(drop_keys)

        assert keys == set(out_item.keys())

    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("prob", [[0, 1], [1, 0]])
    @pytest.mark.parametrize("functions", itertools.permutations([
        {"type": "skip"},
        {"type": "drop", "arguments": {"keys": ["sequence"]}},
        {"type": "whitespace_corruption"},
        {"type": "edit_tokens_corruption"}
    ], r=2))
    def test_switch(self, functions: List[Dict[str, Any]], prob: List[float], seed: int) -> None:
        with pytest.raises(AssertionError):
            _ = get_preprocessing_fn("switch",
                                     functions=[],
                                     prob=[0.1],
                                     seed=seed)

        with pytest.raises(AssertionError):
            _ = get_preprocessing_fn("switch",
                                     functions=[get_preprocessing_fn("skip")],
                                     prob=[0.8],
                                     seed=seed)

        preprocessing_fn = get_preprocessing_fn("switch",
                                                functions=functions,
                                                prob=prob,
                                                seed=seed)

        test_item = {"sequence": "This is a test"}

        out_item = preprocessing_fn(test_item)

        fn = functions[prob.index(1)]
        p_fn = get_preprocessing_fn(fn["type"], **fn.get("arguments", {}))
        assert out_item == p_fn(test_item)

    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("tokenizer", ["byte", "char"])
    def test_character_masked_language_modeling(self, seed: int, tokenizer: str) -> None:
        preprocessing_fn = get_preprocessing_fn("character_masked_language_modeling",
                                                seed=seed,
                                                word_p=1.0,
                                                full_word_p=0.2)

        test_item = {"sequence": "This is a test sentence to test masked language modeling"}

        out_item = preprocessing_fn(test_item)

        assert "labels" in out_item
        assert constants.MASK in out_item["sequence"]

    @pytest.mark.parametrize("target_first", [True, False])
    def test_join_sequences(self, target_first: bool) -> None:
        preprocessing_fn = get_preprocessing_fn("join_sequences", target_first=target_first)

        test_item = {"sequence": "This is a test"}

        with pytest.raises(AssertionError):
            preprocessing_fn(test_item)

        test_item["target_sequence"] = "This is also a test"

        out_item = preprocessing_fn(test_item)

        assert "target_sequence" not in out_item
        assert constants.SEP in out_item["sequence"]
        assert len(out_item["sequence"]) == \
               len(test_item["sequence"]) + len(constants.SEP) + 2 + len(test_item["target_sequence"])
        if target_first:
            assert out_item["sequence"].index("also") < len(out_item["sequence"]) // 2
        else:
            assert out_item["sequence"].index("also") > len(out_item["sequence"]) // 2

    def test_edit_distance(self) -> None:
        preprocessing_fn = get_preprocessing_fn("edit_distance", tokenizer="char")

        test_item = {"sequence": "This is a test"}

        with pytest.raises(AssertionError):
            preprocessing_fn(test_item)

        test_item["target_sequence"] = "This is o tist"

        out_item = preprocessing_fn(test_item)

        assert "target_sequence" not in out_item
        assert "labels" in out_item
        assert out_item["labels"] == 2

    def test_edit_operations(self) -> None:
        preprocessing_fn = get_preprocessing_fn("edit_operations", tokenizer="char")

        test_item = {"sequence": "This is a test"}

        with pytest.raises(AssertionError):
            preprocessing_fn(test_item)

        test_item["target_sequence"] = "This is o tist"

        out_item = preprocessing_fn(test_item)

        assert "target_sequence" not in out_item
        assert "labels" in out_item
        assert out_item["labels"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_edit_tokens_corruption(self, seed: int) -> None:
        preprocessing_fn = get_preprocessing_fn("edit_tokens_corruption", p=0.5, seed=seed)

        test_item = {"sequence": "This is a test sentence to test edit tokens corruption"}

        out_item = preprocessing_fn(test_item)

        assert out_item["target_sequence"] == test_item["sequence"]
        assert out_item["sequence"] != test_item["sequence"]

    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("use_labels", [True, False])
    def test_tokenization_repair_corruption(self, seed: int, use_labels: bool) -> None:
        preprocessing_fn = get_preprocessing_fn("tokenization_repair_corruption",
                                                iw_p=0.5,
                                                dw_p=0.5,
                                                use_labels=use_labels,
                                                seed=seed)

        test_item = {"sequence": "This is a test sentence to test tokenization repair corruption"}

        out_item = preprocessing_fn(test_item)

        assert set(test_item["sequence"]) - set(out_item["sequence"]) <= {" "}
        if use_labels:
            assert "labels" in out_item
            assert "target_sequence" not in out_item
            assert set(out_item["labels"]) <= {0, 1, 2}
            assert len(out_item["labels"]) == len(out_item["sequence"]) + 2
        else:
            assert "labels" not in out_item
            assert "target_sequence" in out_item
            assert set(out_item["target_sequence"]) <= {TokenizationRepairTokens.INSERT_WS.value,
                                                        TokenizationRepairTokens.DELETE_WS.value,
                                                        TokenizationRepairTokens.KEEP_CHAR.value}
            assert len(out_item["target_sequence"]) == len(out_item["sequence"])

    @pytest.mark.parametrize("output_type", ["repair_token", "char", "label"])
    def test_tokenization_repair(self, output_type: str) -> None:
        preprocessing_fn = get_preprocessing_fn("tokenization_repair", output_type=output_type)

        test_item = {"sequence": "Thi s is atest"}

        with pytest.raises(AssertionError):
            _ = preprocessing_fn(test_item)

        test_item["target_sequence"] = "This is a test"

        out_item = preprocessing_fn(test_item)

        assert set(test_item["sequence"]) - set(out_item["sequence"]) <= {" "}
        if output_type == "label":
            assert "labels" in out_item
            assert "target_sequence" not in out_item
            assert out_item["labels"] == [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif output_type == "repair_token":
            assert "labels" not in out_item
            assert "target_sequence" in out_item
            assert out_item["target_sequence"] == "###x######_###"
        else:
            assert "labels" not in out_item
            assert "target_sequence" in out_item
            assert out_item["target_sequence"] == "This is a test"
