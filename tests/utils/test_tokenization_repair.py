import string
import sys
sys.path.append("..")

import pytest

from conftest import randomly_delete_whitespaces, randomly_insert_whitespaces

from whitespace_repair.utils import tokenization_repair


class TestTokenizationRepair:

    @pytest.mark.parametrize("execution", list(range(100)))
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_remove_whitespace(self, execution: int, seed: int) -> None:
        s = string.ascii_letters
        new_s = randomly_insert_whitespaces(s, p=0.5, seed=seed)

        assert len(new_s) > len(s)
        assert set(new_s) - set(s) == {" "}
        assert s == tokenization_repair.remove_whitespace(new_s)

    @pytest.mark.parametrize("execution", list(range(100)))
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_get_whitespace_operations_and_repair_whitespace(self, execution: int, seed: int) -> None:
        s = randomly_insert_whitespaces(string.ascii_letters, p=0.2, seed=seed)
        new_s = randomly_delete_whitespaces(s, p=0.5, seed=seed)

        assert set(s) - set(new_s) <= {" "}

        s_to_new_s_ops = tokenization_repair.get_whitespace_operations(s, new_s)
        assert new_s == tokenization_repair.repair_whitespace(s, s_to_new_s_ops)

        new_s_to_s_ops = tokenization_repair.get_whitespace_operations(new_s, s)
        assert s == tokenization_repair.repair_whitespace(new_s, new_s_to_s_ops)
