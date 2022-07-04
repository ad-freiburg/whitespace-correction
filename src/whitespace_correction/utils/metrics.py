import collections
import math
from abc import ABC
from typing import Any, Callable, List, Optional, Set, Tuple, Union, no_type_check

from editdistance import distance as ed

import numpy as np

import tokenizers

import torch

from whitespace_correction.utils import constants, whitespace_correction


def _ed(l1: Union[List[str], List[List]], l2: Union[List[str], List[List]]) -> float:
    eds = [ed(s, t) for s, t in zip(l1, l2)]
    return sum(eds) / max(len(eds), 1)


def _ned(l1: Union[List[str], List[List]], l2: Union[List[str], List[List]]) -> float:
    ned = [
        ed(s, t) / max(len(s), len(t))
        for s, t in zip(l1, l2)
    ]
    return sum(ned) / max(len(ned), 1)


def mean_sequence_edit_distance(sequences: List[str], target_sequences: List[str]) -> float:
    return _ed(sequences, target_sequences)


def token_edit_distance(input_ids: List[List[int]],
                        target_input_ids: List[List[int]]) -> float:
    return _ed(input_ids, target_input_ids)


def mean_normalized_sequence_edit_distance(sequences: List[str], target_sequences: List[str]) -> float:
    """

    Normalized edit distance on strings:
        ED(A, B) / max(len(A), len(B)) with A and B being strings

    :param sequences: list of strings
    :param target_sequences: list of strings
    :return: mean distance over all pairs
    """
    return _ned(sequences, target_sequences)


def normalized_token_edit_distance(input_ids: List[List[int]],
                                   target_input_ids: List[List[int]]) -> float:
    """

    Normalized edit distance on tokens:
        ED(A, B) / max(len(A), len(B)) with A and B being token lists
    Very similar to Word Error Rate (WER) when computed on word tokens:
        ED(A, B) / len(B) with A and B being word token ids and B being the target reference
        (see wikipedia https://en.wikipedia.org/wiki/Word_error_rate)

    :param input_ids: list of token lists
    :param target_input_ids: list of token lists
    :return: mean distance over all pairs
    """
    return _ned(input_ids, target_input_ids)


def sequence_accuracy(sequences: Union[List[str], torch.Tensor],
                      target_sequences: Union[List[str], torch.Tensor]) -> float:
    """

    What percentage out of the given sequences match the target sequences:
        1 if A == B else 0 with A and B being strings or list of tokens

    :param sequences: list of strings or tensor
    :param target_sequences: list of strings or tensor
    :return: mean accuracy over all pairs
    """
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.tolist()
    if isinstance(target_sequences, torch.Tensor):
        target_sequences = target_sequences.tolist()
    equal = [
        sequence == target_sequence
        for sequence, target_sequence in zip(sequences, target_sequences)
    ]
    return sum(equal) / max(len(equal), 1)


def _tp_fp_fn_to_f1_prec_rec(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # when there are no true positives, precision and recall are zero and f1 is undefined
    if tp == 0:
        return None, None, None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def _insertions_and_deletions(repair_ops: List[int]) -> Set[Tuple[int, int]]:
    insertions_and_deletions = set()
    for i, op in enumerate(repair_ops):
        if op != 0:
            insertions_and_deletions.add((i, op))
    return insertions_and_deletions


def whitespace_correction_f1_prec_rec(
        sequences: List[str],
        target_sequences: List[str],
        input_sequences: List[str],
        mode: str = "insertions_and_deletions"
) -> Tuple[float, float, float, float, float, float]:
    assert mode in {"insertions_and_deletions", "insertions", "deletions"}
    tp = 0
    fp = 0
    fn = 0

    f1s = []
    precs = []
    recs = []

    for seq, gt, ipt in zip(sequences, target_sequences, input_sequences):
        gt_ops = whitespace_correction.get_whitespace_operations(ipt, gt)
        pred_ops = whitespace_correction.get_whitespace_operations(ipt, seq)
        assert len(gt_ops) == len(pred_ops)

        gt_insertions_and_deletions = _insertions_and_deletions(gt_ops)
        pred_insertions_and_deletions = _insertions_and_deletions(pred_ops)

        if mode == "insertions":
            gt_insertions_and_deletions = set(filter(lambda e: e[1] == 1, gt_insertions_and_deletions))
            pred_insertions_and_deletions = set(filter(lambda e: e[1] == 1, pred_insertions_and_deletions))
        elif mode == "deletions":
            gt_insertions_and_deletions = set(filter(lambda e: e[1] == 2, gt_insertions_and_deletions))
            pred_insertions_and_deletions = set(filter(lambda e: e[1] == 2, pred_insertions_and_deletions))

        tp_ = len(gt_insertions_and_deletions.intersection(pred_insertions_and_deletions))
        fp_ = len(pred_insertions_and_deletions.difference(gt_insertions_and_deletions))
        fn_ = len(gt_insertions_and_deletions.difference(pred_insertions_and_deletions))

        tp += tp_
        fp += fp_
        fn += fn_

        scores = _tp_fp_fn_to_f1_prec_rec(tp_, fp_, fn_)
        if all(s is None for s in scores):
            # scores are none if either
            # 1. there are no groundtruth operations (tp == fp == fn == 0)
            # 2. there are no true positives (tp == 0)
            if len(gt_insertions_and_deletions) == 0 and len(pred_insertions_and_deletions) == 0:
                scores = (1, 1, 1)
            else:
                assert tp_ == 0
                scores = (0, 0, 0)

        f1, prec, rec = scores
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    f1_seq, prec_seq, rec_seq = np.mean(f1s) if f1s else 0, np.mean(precs) if precs else 0, np.mean(recs) if recs else 0
    f1_mic, prec_mic, rec_mic = _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)

    return f1_mic or 0, prec_mic or 0, rec_mic or 0, f1_seq, prec_seq, rec_seq


def f1_prec_rec(sequences: List[str],
                target_sequences: List[str],
                split_fn: Optional[Callable] = None) -> Tuple[float, float, float]:
    if split_fn is None:
        def _split_fn(s: str) -> List[str]:
            return s.split(" ")

        split_fn = _split_fn

    tp = 0
    fp = 0
    fn = 0

    for sequence, target_sequence in zip(sequences, target_sequences):
        tokens = split_fn(sequence)
        target_tokens = split_fn(target_sequence)

        tokens_count = collections.Counter(tokens)
        target_tokens_count = collections.Counter(target_tokens)

        tokens_in_common = sum((tokens_count & target_tokens_count).values())

        tp += tokens_in_common
        fp += len(tokens) - tokens_in_common
        fn += len(target_tokens) - tokens_in_common

    return _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)


class Metric(ABC):
    @no_type_check
    def add(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def calc(self) -> Any:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()


class TextMetric(Metric, ABC):
    def __init__(self, with_special_tokens: bool = False):
        super().__init__()
        self.inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.outputs = None
        self.labels = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.with_special_tokens = with_special_tokens

    def add(self,
            inputs: Tuple[torch.Tensor, ...],
            outputs: torch.Tensor,
            labels: torch.Tensor,
            encoder_tokenizer: tokenizers.Tokenizer,
            decoder_tokenizer: tokenizers.Tokenizer = None,
            **kwargs: Any) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

    def calc(self) -> str:
        raise NotImplementedError()

    def reset(self) -> None:
        self.inputs = None
        self.outputs = None
        self.labels = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None


class AverageMetric(Metric):
    def __init__(self, name: str):
        self._name = name
        self.num_adds = 0
        self.sum = 0.0

    def add(self, x: Union[torch.Tensor, float]) -> None:
        self.sum += x.item() if isinstance(x, torch.Tensor) else x
        self.num_adds += 1

    def calc(self) -> float:
        return self.sum / self.num_adds if self.num_adds > 0 else 0

    def reset(self) -> None:
        self.num_adds = 0
        self.sum = 0.0

    def name(self) -> str:
        return self._name


class Perplexity(Metric):
    def __init__(self) -> None:
        self.normalizer = 0.0
        self.neg_log_prob_sum = 0.0

    def add(self, x: torch.Tensor) -> None:
        log_probabilities = torch.max(torch.log_softmax(x, dim=1), dim=1)[0]
        self.neg_log_prob_sum += -torch.sum(log_probabilities).item()
        self.normalizer += len(log_probabilities)

    def calc(self) -> float:
        return math.exp(self.neg_log_prob_sum / self.normalizer)

    def reset(self) -> None:
        self.normalizer = 0.0
        self.neg_log_prob_sum = 0.0

    def name(self) -> str:
        return "perplexity"


def get_text_metric(name: str, **kwargs: Any) -> TextMetric:
    if name == QualitativeBatchEvaluation().name():
        return QualitativeBatchEvaluation(**kwargs)
    elif name == QualitativeBatchEvaluationWhitespaceCorrection().name():
        return QualitativeBatchEvaluationWhitespaceCorrection(**kwargs)
    elif name == QualitativeBatchEvaluationClassification().name():
        return QualitativeBatchEvaluationClassification(**kwargs)
    elif name == QualitativeBatchEvaluationSequenceClassification().name():
        return QualitativeBatchEvaluationSequenceClassification(**kwargs)
    else:
        raise ValueError(f"Unknown text metric {name}")


class QualitativeBatchEvaluation(TextMetric):
    def calc(self) -> str:
        assert self.inputs is not None and self.outputs is not None and self.labels is not None
        assert len(self.inputs) in {1, 2}
        if len(self.inputs) == 1:
            input_ids, target_ids = self.inputs[0], None
        else:
            input_ids, target_ids = self.inputs

        input_str = self.encoder_tokenizer.decode_batch(input_ids.T.tolist(),
                                                        skip_special_tokens=not self.with_special_tokens)

        pred_ids = torch.argmax(self.outputs, dim=2).T.tolist()
        pred_str = self.decoder_tokenizer.decode_batch(pred_ids,
                                                       skip_special_tokens=not self.with_special_tokens)

        if target_ids is not None:
            target_str = self.decoder_tokenizer.decode_batch(target_ids.T.tolist(),
                                                             skip_special_tokens=not self.with_special_tokens)

        else:
            target_str = self.decoder_tokenizer.decode_batch(self.labels.tolist(),
                                                             skip_special_tokens=not self.with_special_tokens)

        B = len(input_str)

        s = ""
        for i in range(B):
            s += f"\n\nInput: {input_str[i]}" \
                 f"\n\nPredicted: {pred_str[i]}" \
                 f"\n\n(Target: {target_str[i]})\n\n"
            if (i + 1) < B:
                s += "-" * 80
        return s

    def name(self) -> str:
        return "qualitative_batch_evaluation"


class QualitativeBatchEvaluationWhitespaceCorrection(QualitativeBatchEvaluation):

    def calc(self) -> str:
        assert self.inputs is not None and self.outputs is not None and self.labels is not None
        assert len(self.inputs) in {1, 2}
        if len(self.inputs) == 1:
            input_ids, target_ids = self.inputs[0], None
        else:
            input_ids, target_ids = self.inputs

        hashtag_token_id = self.encoder_tokenizer.token_to_id("#")
        unk_token_id = self.encoder_tokenizer.token_to_id(constants.UNK)
        # replace unk tokens with single char hashtags, because <unk> would not line up with
        # the one label per character rule of tokenization repair
        input_ids[input_ids == unk_token_id] = hashtag_token_id

        input_str = self.encoder_tokenizer.decode_batch(input_ids.T.tolist())

        if target_ids is not None:
            target_str = self.decoder_tokenizer.decode_batch(target_ids.T.tolist())
            repaired_target_str = [whitespace_correction.repair_whitespace(in_s, t_s)
                                   for in_s, t_s in zip(input_str, target_str)]

            pred_ids = torch.argmax(self.outputs, dim=2).T.tolist()
            pred_str = self.decoder_tokenizer.decode_batch(pred_ids)
            repaired_pred_str = [whitespace_correction.repair_whitespace(in_s, p_s)
                                 for in_s, p_s in zip(input_str, pred_str)]

        else:
            target_str = self.labels[:, 1:-1].tolist()
            repaired_target_str = [whitespace_correction.repair_whitespace(in_s, t_s)
                                   for in_s, t_s in zip(input_str, target_str)]

            pred_ids = torch.argmax(self.outputs, dim=2).T[:, 1:-1].tolist()
            pred_str = pred_ids
            repaired_pred_str = [whitespace_correction.repair_whitespace(in_s, p_s)
                                 for in_s, p_s in zip(input_str, pred_str)]

        B = len(input_str)

        s = ""
        for i in range(B):
            s += f"\n\nInput: {input_str[i]}" \
                 f"\n\nPredicted: {repaired_pred_str[i]}\n(Repair tokens: {pred_str[i]})" \
                 f"\n\n(Target: {repaired_target_str[i]})\n(Target repair tokens: {target_str[i]})\n\n"
            if (i + 1) < B:
                s += "-" * 80
        return s

    def name(self) -> str:
        return "qualitative_batch_evaluation_whitespace_correction"


class QualitativeBatchEvaluationClassification(TextMetric):
    def calc(self) -> str:
        assert self.inputs is not None and self.outputs is not None and self.labels is not None
        input_ids = self.inputs[0]
        B, _ = self.outputs.shape
        predicted_class = torch.argmax(self.outputs, dim=1).T.tolist()
        input_str = self.encoder_tokenizer.decode_batch(input_ids.T.tolist(),
                                                        skip_special_tokens=not self.with_special_tokens)
        labels = self.labels.tolist()

        s = ""
        for i in range(B):
            s += f"\n\nInput: {input_str[i]}" \
                 f"\n\nPredicted class: {predicted_class[i]}" \
                 f"\n\n(Target class: {labels[i]})\n\n"
            if (i + 1) < B:
                s += "-" * 80
        return s

    def name(self) -> str:
        return "qualitative_batch_evaluation_classification"


class QualitativeBatchEvaluationSequenceClassification(TextMetric):
    def calc(self) -> str:
        assert self.inputs is not None and self.outputs is not None and self.labels is not None
        input_ids = self.inputs[0]
        S, B, _ = self.outputs.shape
        predicted_classes = torch.argmax(self.outputs, dim=2).T.tolist()
        input_str = self.encoder_tokenizer.decode_batch(input_ids.T.tolist(),
                                                        skip_special_tokens=not self.with_special_tokens)
        labels = self.labels.tolist()

        s = ""
        for i in range(B):
            s += f"\n\nInput: {input_str[i]}" \
                 f"\n\nPredicted classes: {predicted_classes[i]}" \
                 f"\n\n(Target classes: {labels[i]})\n\n"
            if (i + 1) < B:
                s += "-" * 80
        return s

    def name(self) -> str:
        return "qualitative_batch_evaluation_sequence_classification"
