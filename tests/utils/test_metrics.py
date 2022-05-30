import itertools
from typing import List, Tuple

import pytest

from whitespace_repair.utils import metrics


class TestMetrics:
    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ("This isa test", "This is a test", 1),
                                                ("Hw areyou", "How are you?", 3),
                                                ("Cool tst", "Cool test", 1),
                                                ("abcdefgh", "ijklmnop", 8)
                                            ], r=i)])
    def test_sequence_edit_distance(self, inputs: List[Tuple[str, str, int]]) -> None:
        expected_ed = sum(i[2] for i in inputs) / len(inputs)

        ed = metrics.mean_sequence_edit_distance(sequences=[i[0] for i in inputs],
                                                 target_sequences=[i[1] for i in inputs])
        assert ed == expected_ed

    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ([0, 5, 6, 3, 2], [0, 5, 6, 2], 1),
                                                ([1, 7, 7, 7, 8], [2, 4, 7, 7, 8, 7], 3),
                                                ([4, 2], [2, 4], 2),
                                                ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], 7)
                                            ], r=i)])
    def test_token_edit_distance(self, inputs: List[Tuple[List[int], List[int], int]]) -> None:
        expected_ed = sum(i[2] for i in inputs) / len(inputs)

        ed = metrics.token_edit_distance(input_ids=[i[0] for i in inputs],
                                         target_input_ids=[i[1] for i in inputs])
        assert ed == expected_ed

    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ("This isa test", "This is a test", 1 / 14),
                                                ("Hw areyou", "How are you?", 3 / 12),
                                                ("Cool tst", "Cool test", 1 / 9),
                                                ("abcdefgh", "ijklmnop", 8 / 8)
                                            ], r=i)])
    def test_normalized_sequence_edit_distance(self, inputs: List[Tuple[str, str, float]]) -> None:
        expected_ned = sum(i[2] for i in inputs) / len(inputs)

        ned = metrics.mean_normalized_sequence_edit_distance(sequences=[i[0] for i in inputs],
                                                             target_sequences=[i[1] for i in inputs])
        assert ned == expected_ned

    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ([0, 5, 6, 3, 2], [0, 5, 6, 2], 1 / 5),
                                                ([1, 7, 7, 7, 8], [2, 4, 7, 7, 8, 7], 3 / 6),
                                                ([4, 2], [2, 4], 2 / 2),
                                                ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], 7 / 7)
                                            ], r=i)])
    def test_normalized_token_edit_distance(self, inputs: List[Tuple[List[int], List[int], float]]) -> None:
        expected_ned = sum(i[2] for i in inputs) / len(inputs)

        ned = metrics.normalized_token_edit_distance(input_ids=[i[0] for i in inputs],
                                                     target_input_ids=[i[1] for i in inputs])
        assert ned == expected_ned

    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ("This isa test", "This is a test"),
                                                ("Hw are you?", "How are you?"),
                                                ("Cool tost", "Cool test"),
                                                ("abcdefgh", "abcdefgh")
                                            ], r=i)])
    def test_sequence_accuracy(self, inputs: List[Tuple[str, str]]) -> None:
        expected_seq_acc = sum(i[0] == i[1] for i in inputs) / len(inputs)

        seq_acc = metrics.sequence_accuracy(sequences=[i[0] for i in inputs],
                                            target_sequences=[i[1] for i in inputs])

        assert seq_acc == expected_seq_acc

    @pytest.mark.parametrize("inputs", [perm
                                        for i in range(1, 4)
                                        for perm in
                                        itertools.permutations(
                                            [
                                                ("This isa test", "This is a test", 2, 1, 2),
                                                ("Hw are you?", "How are you?", 2, 1, 1),
                                                ("Cool tost", "Cool test", 1, 1, 1),
                                                ("abcdefgh", "abcdefgh", 1, 0, 0)
                                            ], r=i)])
    def test_f1_prec_rec(self, inputs: List[Tuple[str, str, int, int, int]]) -> None:
        tp = sum(i[2] for i in inputs)
        fp = sum(i[3] for i in inputs)
        fn = sum(i[4] for i in inputs)
        expected_prec = tp / (tp + fp)
        expected_rec = tp / (tp + fn)
        expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec)

        f1, prec, rec = metrics.f1_prec_rec(sequences=[i[0] for i in inputs],
                                            target_sequences=[i[1] for i in inputs])

        assert f1 == expected_f1 and prec == expected_prec and rec == expected_rec
