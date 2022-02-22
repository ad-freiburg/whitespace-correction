import random
import string
from typing import List, Tuple

import numpy as np

import pytest

from tests.conftest import randomly_insert_whitespaces

from trt.utils import nlp


class TestNLP:
    @staticmethod
    def add_noise(s: str, p: float, seed: int) -> str:
        rand = random.Random(seed)
        noise_chars = ["\n", "\t"]
        new_s = ""
        for c in s:
            if rand.random() < p and c == " ":
                if rand.random() < 0.33:
                    # multiple whitespaces
                    new_s += rand.randint(1, 5) * " " + c
                else:
                    new_s += rand.choice(noise_chars)
            else:
                new_s += c
        return new_s

    @pytest.mark.parametrize("execution", list(range(100)))
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_clean_sequence(self, execution: int, seed: int) -> None:
        cleaned_sequence = randomly_insert_whitespaces(string.ascii_letters, p=0.2, seed=seed).strip()
        uncleaned_sequence = TestNLP.add_noise(cleaned_sequence, p=0.2, seed=seed)

        assert nlp.clean_sequence(uncleaned_sequence) == cleaned_sequence

    @pytest.mark.parametrize("sequence", [
        ("This is a valid sequence", True),
        ("He was born 1992", True),
        ("123899213", False),
        ("This <html> is an invalid sequence", False),
        (", .. 1++", False)
    ])
    @pytest.mark.parametrize("min_length", [0, 5, 10])
    def test_is_valid_sequence(self, sequence: Tuple[str, int], min_length: int) -> None:
        assert nlp.is_valid_sequence(sequence[0], min_length=min_length) == \
               (sequence[1] if len(sequence[0]) >= min_length else False)

    @pytest.mark.parametrize("tokens", [
        (["This", "is", "a", "test"], "This is a test"),
        (["I", "have", "n't", "done", "this"], "I haven't done this"),
        (["Sentence", "with", "punctuation", "!"], "Sentence with punctuation!"),
    ])
    def test_tokens_to_text(self, tokens: List[str]) -> None:
        assert nlp.tokens_to_text(tokens[0]) == tokens[1]

    @pytest.mark.parametrize("token", ["Hello", "Test", "something"])
    @pytest.mark.parametrize("include", [
        [0],
        [1],
        [2],
        [3],
        [0, 1, 2, 3]
    ])
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_edit_token(self, token: str, include: List[int], seed: int) -> None:
        edited_token = nlp.edit_token(token, include=include, rand=np.random.RandomState(seed))
        if include == 0:
            # insert
            assert len(edited_token) == len(token) + 1
        elif include == 1:
            # delete
            assert len(edited_token) == len(token) - 1
        elif include == 2:
            # swap
            assert len(token) == len(edited_token)
            assert set(token) - set(edited_token) == {}
        elif include == 3:
            # replace
            assert len(token) == len(edited_token)
            assert len(set(token) - set(edited_token)) == 1

        assert edited_token != token
