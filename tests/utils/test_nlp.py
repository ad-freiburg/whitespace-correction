import random
import string
from typing import Tuple
import sys

sys.path.append("..")

import numpy as np
import pytest

from whitespace_repair.utils import nlp

from conftest import randomly_insert_whitespaces


class TestNLP:
    @staticmethod
    def hamming_dist(a: str, b: str) -> int:
        assert len(a) == len(b)
        return sum(ac != bc for ac, bc in zip(a, b))

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

    @pytest.mark.parametrize("token", ["Hello", "Test", "something"])
    @pytest.mark.parametrize("include", [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1, 2, 3)
    ])
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_edit_token(self, token: str, include: Tuple[int], seed: int) -> None:
        edited_token, edited_indices = nlp.edit_token(token, include=include, rand=np.random.RandomState(seed))
        if include == (0,):
            # insert
            assert len(edited_token) == len(token) + 1, f"{edited_token} and {token} should differ 1 in length"
        elif include == (1,):
            # delete
            assert len(edited_token) == len(token) - 1, f"{edited_token} and {token} should differ 1 in length"
        elif include == (2,):
            # swap
            assert len(token) == len(edited_token), f"{edited_token} and {token} have different lengths"
            assert len(set(token).symmetric_difference(set(edited_token))) == 0, \
                f"{edited_token} and {token} contain different chars"
        elif include == (3,):
            # replace
            assert len(token) == len(edited_token), f"{edited_token} and {token} have different lengths"
            assert TestNLP.hamming_dist(token, edited_token) == 1, \
                f"{edited_token} and {token} should differ in 1 char"
