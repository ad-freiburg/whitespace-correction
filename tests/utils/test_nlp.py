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
        edited_token = nlp.edit_token(token, include=include, rand=np.random.RandomState(seed))
        if include == (0, ):
            # insert
            assert len(edited_token) == len(token) + 1
        elif include == (1, ):
            # delete
            assert len(edited_token) == len(token) - 1
        elif include == (2, ):
            # swap
            assert len(token) == len(edited_token)
            assert set(token) - set(edited_token) == {}
        elif include == (3, ):
            # replace
            assert len(token) == len(edited_token)
            assert len(set(token) - set(edited_token)) == 1

        assert edited_token != token
