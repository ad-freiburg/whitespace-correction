import random
from typing import Any

import pytest

from whitespace_correction.utils.config import PreprocessingConfig


@pytest.fixture
def preprocessing_config(request: Any) -> PreprocessingConfig:
    return PreprocessingConfig(type=request.param,
                               arguments={})


def randomly_insert_whitespaces(s: str, p: float, seed: int) -> str:
    rand = random.Random(seed)
    new_s = ""
    for i, c in enumerate(s):
        if rand.random() < p and c != " " and (s[i - 1] != " " if i > 0 else True):
            new_s += " " + c
        else:
            new_s += c
    return new_s


def randomly_delete_whitespaces(s: str, p: float, seed: int) -> str:
    rand = random.Random(seed)
    new_s = ""
    for c in s:
        if rand.random() < p and c == " ":
            continue
        else:
            new_s += c
    return new_s
