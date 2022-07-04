import queue
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import tokenizers

import torch
from torch import nn
from torch.nn.utils import rnn

from whitespace_correction.utils import common, constants

logger = common.get_logger("INFERENCE")


@dataclass
class InferenceResult:
    pass


@dataclass
class ClassificationInferenceResult(InferenceResult):
    prediction: int
    logits: List[float] = None


@dataclass
class SequenceClassificationInferenceResult(InferenceResult):
    predictions: List[int]
    logits: List[List[float]] = None

    @property
    def length(self) -> int:
        return len(self.predictions)


@dataclass
class SequenceGenerationInferenceResult(InferenceResult):
    token_ids: List[int]
    token_log_probabilities: List[float]

    @property
    def sequence_log_probability(self) -> float:
        return sum(self.token_log_probabilities)

    @property
    def length(self) -> int:
        return len(self.token_ids)


class Beam:
    def __init__(self,
                 log_probabilities: List[float],
                 token_ids: List[int],
                 eos: bool):
        self.log_probabilities = log_probabilities
        self.token_ids = token_ids
        self.eos = eos

    def get_decoder_input(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.token_ids, device=device)


DeTokFn = Callable[[List[int]], str]
SelectFn = Callable[[List[Tuple[Beam, float]]], Beam]
ScoreFn = Callable[[Beam, Optional[str]], float]


def log_likelihood_score_fn(normalize_by_length: bool = True, alpha: float = 1.0) -> ScoreFn:
    def _log_l(beam: Beam, _: str = None) -> float:
        s = sum(beam.log_probabilities)
        if normalize_by_length:
            s /= (len(beam.log_probabilities) ** alpha)
        return s

    return _log_l


def greedy_select_fn() -> SelectFn:
    def _greedy(beams: List[Tuple[Beam, float]]) -> Beam:
        return beams[0][0]

    return _greedy


def sample_select_fn(sample_top_k: int) -> SelectFn:
    def _greedy(beams: List[Tuple[Beam, float]]) -> Beam:
        sample_idx = torch.randint(min(len(beams), sample_top_k)).item()
        return beams[sample_idx][0]

    return _greedy


def get_temperatures_thresholds_and_defaults(
        no_spaces: List[bool],
        temperature: float,
        temperature_no_spaces: float,
        thresholds_and_default: Optional[Tuple[Tuple[float, ...], int]] = None,
        thresholds_and_default_no_spaces: Optional[Tuple[Tuple[float, ...], int]] = None,
        **_: Any
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    temperatures = torch.tensor(
        [temperature_no_spaces if no_space else temperature for no_space in no_spaces], dtype=torch.float
    )
    if thresholds_and_default is not None and thresholds_and_default_no_spaces is not None:
        thresholds, default = thresholds_and_default
        thresholds_no_spaces, default_no_spaces = thresholds_and_default_no_spaces
        thresholds_and_defaults = (
            torch.tensor(
                [thresholds_no_spaces if no_space else thresholds for no_space in no_spaces], dtype=torch.float
            ),
            torch.tensor(
                [default_no_spaces if no_space else default for no_space in no_spaces], dtype=torch.long
            )
        )
    else:
        thresholds_and_defaults = None
    return temperatures, thresholds_and_defaults


def class_predictions_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        thresholds_and_defaults: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> List:
    assert len(temperatures) == len(logits)
    b = logits.shape[0]
    c = logits.shape[-1]
    intermediate = logits.shape[1:-1]
    repeat = int(np.prod(intermediate))

    temperatures = torch.repeat_interleave(temperatures, repeat, 0)
    probabilities = torch.softmax(logits.cpu().reshape(-1, c) / temperatures.unsqueeze(1), -1)
    best_probabilities, best_indices = torch.max(probabilities, -1)

    if thresholds_and_defaults is None:
        return best_indices.reshape(b, *intermediate).tolist()
    else:
        assert all(len(i) == len(logits) for i in thresholds_and_defaults)
        num_classes = logits.shape[-1]
        thresholds, default = thresholds_and_defaults
        thresholds = torch.repeat_interleave(thresholds, repeat, 0)
        default = torch.repeat_interleave(default, repeat, 0)
        assert thresholds.shape[-1] == num_classes and torch.all(torch.logical_and(num_classes > default, default >= 0))
        return torch.where(
            best_probabilities >= thresholds[torch.arange(len(thresholds)), best_indices],
            best_indices,
            default
        ).reshape(b, *intermediate).tolist()


_INFERENCE_METHODS = {"greedy", "sample", "beam"}


def sequences_to_ids(sequences: Union[str, List[str]],
                     tokenizer: tokenizers.Tokenizer,
                     device: torch.device,
                     as_list: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
    if not isinstance(sequences, list):
        sequences = [sequences]

    encodings = tokenizer.encode_batch(sequences)
    input_ids = [torch.tensor(enc.ids, device=device, dtype=torch.long) for enc in encodings]
    if as_list:
        return input_ids

    padded_input_ids = rnn.pad_sequence(input_ids,
                                        batch_first=True,
                                        padding_value=tokenizer.token_to_id(constants.PAD))
    return padded_input_ids


def inference_with_ids(
        model: nn.Module,
        bos_token_id: int,
        eos_token_id: int,
        max_input_length: int,
        max_output_length: int,
        device: torch.device,
        method: str,
        decoder_only: bool = False,
        score_fn: ScoreFn = log_likelihood_score_fn(),
        input_strings: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[List[torch.Tensor]] = None,
        decoder_padding_token_id: Optional[int] = None,
        **kwargs: Any
) -> Union[List[List[SequenceGenerationInferenceResult]], List[SequenceGenerationInferenceResult]]:
    if method == "greedy":
        return batch_inference(model=model,
                               bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_input_length=max_input_length,
                               max_output_length=max_output_length,
                               device=device,
                               select_fn=greedy_select_fn(),
                               score_fn=score_fn,
                               input_strings=input_strings,
                               input_ids=input_ids,
                               decoder_only=decoder_only,
                               decoder_input_ids=decoder_input_ids,
                               decoder_padding_token_id=decoder_padding_token_id)
    elif method == "sample":
        return batch_inference(model=model,
                               bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_input_length=max_input_length,
                               max_output_length=max_output_length,
                               device=device,
                               select_fn=sample_select_fn(kwargs.get("sample_top_k", 3)),
                               score_fn=score_fn,
                               input_strings=input_strings,
                               input_ids=input_ids,
                               decoder_only=decoder_only,
                               decoder_input_ids=decoder_input_ids,
                               decoder_padding_token_id=decoder_padding_token_id)
    elif method == "beam":
        return beam_inference(model=model,
                              bos_token_id=bos_token_id,
                              eos_token_id=eos_token_id,
                              max_input_length=max_input_length,
                              max_output_length=max_output_length,
                              device=device,
                              score_fn=score_fn,
                              input_strings=input_strings,
                              beam_width=kwargs.pop("beam_width", 3),
                              input_ids=input_ids,
                              decoder_only=decoder_only,
                              decoder_input_ids=decoder_input_ids)
    else:
        raise ValueError(f"Unknown inference method {method}. Use one of {_INFERENCE_METHODS}.")


def batch_encode(model: nn.Module,
                 input_ids: torch.Tensor,
                 max_length: int,
                 device: torch.device,
                 return_memory_padding_mask: bool = False) -> torch.Tensor:
    assert hasattr(model, "encode"), "model needs an encode method to be able to used with this function"

    was_training = model.training
    model.to(device)
    model.eval()

    if input_ids.shape[-1] > max_length:
        logger.warning(f"Max length is {max_length}, but got inputs with shape {input_ids.shape}. "
                       f"Cutting them to {max_length}.")
        input_ids = input_ids[..., :max_length]

    with torch.no_grad():
        encoder_outputs, _ = model.encode(src=input_ids.T)

    model.train(was_training)

    if return_memory_padding_mask:
        return encoder_outputs, model.get_memory_key_padding_mask(input_ids.T)

    return encoder_outputs


def batch_autoregressive_decode(model: nn.Module,
                                bos_token_id: int,
                                eos_token_id: int,
                                max_length: int,
                                device: torch.device,
                                select_fn: SelectFn,
                                score_fn: ScoreFn = log_likelihood_score_fn(),
                                input_strings: Optional[List[str]] = None,
                                encoder_outputs: Optional[torch.Tensor] = None,
                                encoder_padding_mask: Optional[torch.Tensor] = None,
                                decoder_input_ids: Optional[List[torch.Tensor]] = None,
                                decoder_padding_token_id: Optional[int] = None,
                                decoder_only: bool = False) -> List[SequenceGenerationInferenceResult]:
    assert hasattr(model, "decode"), "model needs a decode method to be able to used with this function"

    if decoder_only:
        assert encoder_outputs is None and decoder_input_ids is not None and decoder_padding_token_id is not None, \
            "in decoder_only mode encoder_outputs must be None and decoder_input_ids and decoder_padding_token_id " \
            "must be not None"

        B = len(decoder_input_ids)
    else:
        assert encoder_outputs is not None
        _, B, _ = encoder_outputs.shape

    was_training = model.training
    model.to(device)
    model.eval()

    log_probs = torch.full((B, max_length), fill_value=-1.0, device=device)
    token_ids = torch.full((B, max_length), fill_value=decoder_padding_token_id if decoder_padding_token_id else -1.0,
                           dtype=torch.long, device=device)

    if decoder_input_ids is not None:
        assert decoder_padding_token_id is not None, "when decoder_input_ids is given, you must also specify " \
                                                     "decoder_padding_token_id"
        lengths = torch.empty(B, dtype=torch.long, device=device)

        for i, input_ids in enumerate(decoder_input_ids):
            assert input_ids[0] == bos_token_id and input_ids[-1] != eos_token_id, \
                "input_ids in decoder_input_ids must start with the bos token and cannot end with the eos token"
            assert len(input_ids) < max_length, "length of input_ids in decoder_input_ids must be smaller " \
                                                "than the max length"
            lengths[i] = len(input_ids)
            token_ids[i, :len(input_ids)] = input_ids.to(device)
            log_probs[i, :len(input_ids)] = 0.0

    else:
        lengths = torch.ones(B, dtype=torch.long, device=device)
        token_ids[:, 0] = bos_token_id
        log_probs[:, 0] = 0.0

    # decode the sequences that dont have an eos token at the end and are still smaller than max length
    # which are all at the beginning of decoding
    non_eos_mask = torch.ones(B, dtype=torch.bool, device=device)
    smaller_max_length_mask = torch.ones(B, dtype=torch.bool, device=device)

    indices_to_decode = non_eos_mask & smaller_max_length_mask

    while True:
        # start = time.perf_counter()
        max_input_length = torch.max(lengths)
        target_input_ids = token_ids[indices_to_decode, :max_input_length]

        encoder_outputs_i = encoder_outputs[:, indices_to_decode, :] if encoder_outputs is not None else None
        encoder_padding_mask_i = encoder_padding_mask[indices_to_decode, :] \
            if encoder_padding_mask is not None else None

        with torch.no_grad():
            decoder_output, _ = model.decode(tgt=target_input_ids.T,
                                             memory=encoder_outputs_i,
                                             memory_key_padding_mask=encoder_padding_mask_i)  # S x B x C

        # start_score = time.perf_counter()
        # total_score = 0
        batch_indices = torch.where(indices_to_decode)[0].tolist()
        inferred_token_ids = []
        inferred_log_prob = []
        for i in range(decoder_output.shape[1]):
            length = lengths[indices_to_decode][i]
            log_softmax_scores = torch.log_softmax(decoder_output[length - 1, i], dim=0)

            batch_idx = batch_indices[i]
            beams_and_scores = []
            current_log_probs = log_probs[batch_idx, :length].tolist()
            current_token_ids = token_ids[batch_idx, :length].tolist()
            for token_id, lp in enumerate(log_softmax_scores.tolist()):
                beam = Beam(
                    log_probabilities=current_log_probs + [lp],
                    token_ids=current_token_ids + [token_id],
                    eos=token_id == eos_token_id
                )
                # start_score2 = time.perf_counter()
                beams_and_scores.append(
                    (beam, -score_fn(beam, input_strings[batch_idx] if input_strings is not None else None))
                )
                # end_score2 = time.perf_counter()
                # total_score += (end_score2 - start_score2) * 1000
            beams_and_scores = sorted(beams_and_scores, key=lambda e: e[1])
            selected_beam = select_fn(beams_and_scores)
            inferred_token_ids.append(selected_beam.token_ids[-1])
            inferred_log_prob.append(selected_beam.log_probabilities[-1])

        # end_score = time.perf_counter()
        # print(f"scoring beams2 took {total_score}ms")
        # print(f"scoring beams took {(end_score - start_score) * 1000}ms")

        inferred_token_ids = torch.tensor(
            inferred_token_ids, dtype=torch.long, device=token_ids.device
        )
        token_ids[indices_to_decode, lengths[indices_to_decode]] = inferred_token_ids
        inferred_log_prob = torch.tensor(
            inferred_log_prob, dtype=torch.float, device=log_probs.device
        )
        log_probs[indices_to_decode, lengths[indices_to_decode]] = inferred_log_prob

        lengths[indices_to_decode] += 1

        inferred_eos_indices = torch.where(inferred_token_ids == eos_token_id)[0]
        new_eos_indices = torch.where(indices_to_decode)[0][inferred_eos_indices]
        non_eos_mask[new_eos_indices] = False

        max_length_indices = torch.where(lengths >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        indices_to_decode = non_eos_mask & smaller_max_length_mask

        # all sequences are at max length or all finished with eos token
        if torch.sum(indices_to_decode) == 0:
            break

        # end = time.perf_counter()
        # print(f"one round took {(end - start) * 1000}ms")

    results = []
    token_ids = token_ids.tolist()
    log_probs = log_probs.tolist()

    for i in range(B):
        length = lengths[i]
        tokens = token_ids[i][:length]
        log_probabilities = log_probs[i][:length]

        assert tokens[-1] == eos_token_id or len(tokens) == max_length

        results.append(
            SequenceGenerationInferenceResult(
                token_ids=tokens,
                token_log_probabilities=log_probabilities
            )
        )

    model.train(was_training)
    return results


def _greedy_token_fn() -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def _greedy(decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.max(torch.log_softmax(decoder_output, dim=1), dim=1)
        return indices, values

    return _greedy


def _sample_token_fn(sample_top_k: int) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def _sample(decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k_probabilities, top_k_indices = torch.softmax(decoder_output, dim=1).topk(sample_top_k, dim=1)
        sampled_indices = torch.multinomial(top_k_probabilities, num_samples=1)
        indices = top_k_indices.gather(1, sampled_indices).squeeze(1)
        values = torch.log(top_k_probabilities.gather(1, sampled_indices)).squeeze(1)
        return indices, values

    return _sample


def batch_inference(model: nn.Module,
                    bos_token_id: int,
                    eos_token_id: int,
                    max_input_length: int,
                    max_output_length: int,
                    device: torch.device,
                    select_fn: SelectFn,
                    score_fn: ScoreFn = log_likelihood_score_fn(),
                    input_strings: Optional[List[str]] = None,
                    decoder_only: bool = False,
                    input_ids: Optional[torch.Tensor] = None,
                    decoder_input_ids: Optional[List[torch.Tensor]] = None,
                    decoder_padding_token_id: Optional[int] = None) -> List[SequenceGenerationInferenceResult]:
    if decoder_only:
        assert decoder_input_ids is not None
        return batch_autoregressive_decode(model=model,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           max_length=max_output_length,
                                           select_fn=select_fn,
                                           score_fn=score_fn,
                                           input_strings=input_strings,
                                           encoder_outputs=None,
                                           decoder_input_ids=decoder_input_ids,
                                           decoder_padding_token_id=decoder_padding_token_id,
                                           decoder_only=True,
                                           device=device)
    else:
        assert input_ids is not None
        encoder_outputs, encoder_memory_padding_mask = batch_encode(model=model,
                                                                    input_ids=input_ids,
                                                                    max_length=max_input_length,
                                                                    device=device,
                                                                    return_memory_padding_mask=True)

        return batch_autoregressive_decode(model=model,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           max_length=max_output_length,
                                           select_fn=select_fn,
                                           score_fn=score_fn,
                                           input_strings=input_strings,
                                           encoder_outputs=encoder_outputs,
                                           encoder_padding_mask=encoder_memory_padding_mask,
                                           decoder_input_ids=decoder_input_ids,
                                           decoder_padding_token_id=decoder_padding_token_id,
                                           device=device)


def beam_inference(
        model: nn.Module,
        bos_token_id: int,
        eos_token_id: int,
        max_input_length: int,
        max_output_length: int,
        device: torch.device,
        beam_width: int,
        score_fn: ScoreFn = log_likelihood_score_fn(),
        input_strings: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_only: Optional[bool] = False,
        decoder_input_ids: Optional[List[torch.Tensor]] = None
) -> List[List[SequenceGenerationInferenceResult]]:
    if decoder_only:
        assert decoder_input_ids is not None
        B = len(decoder_input_ids)
    else:
        assert input_ids is not None
        B, _ = input_ids.size()

    model = model.to(device)
    model.eval()

    # 2 * beam_width because half could be eos
    num_candidates = 2 * beam_width

    if decoder_only:
        encoder_outputs, encoder_memory_padding_mask = None, None

    else:
        encoder_outputs, encoder_memory_padding_mask = batch_encode(model=model,
                                                                    input_ids=input_ids,
                                                                    max_length=max_input_length,
                                                                    device=device,
                                                                    return_memory_padding_mask=True)

    all_beams: List[List[Beam]] = []

    for b in range(B):
        beam_queue: queue.PriorityQueue = queue.PriorityQueue()
        # store current beams here, there will be always exactly beam width entries
        if decoder_input_ids is not None:
            decoder_input_ids_i = decoder_input_ids[b]
            current_beams = [
                Beam(log_probabilities=[0.0] * len(decoder_input_ids_i),
                     token_ids=decoder_input_ids_i.tolist(),
                     eos=False)
                for _ in range(beam_width)
            ]

            initial_input_ids = decoder_input_ids_i

            search_depth = len(decoder_input_ids_i)

        else:
            current_beams = [
                Beam(log_probabilities=[0.0], token_ids=[bos_token_id], eos=False)
                for _ in range(beam_width)
            ]

            initial_input_ids = torch.tensor([bos_token_id], device=device)

            search_depth = 1

        # get initial outputs first to initialize the beams
        initial_input_ids = initial_input_ids.unsqueeze(1)

        encoder_outputs_i = encoder_outputs[:, b:b + 1, :] if encoder_outputs is not None else None
        encoder_memory_padding_mask_i = encoder_memory_padding_mask[b:b + 1, :] \
            if encoder_memory_padding_mask is not None else None

        with torch.no_grad():
            initial_decoder_output = model.decode(tgt=initial_input_ids,
                                                  memory=encoder_outputs_i,
                                                  memory_key_padding_mask=encoder_memory_padding_mask_i)[0][-1, 0, :]

        top_k_log_probabilities, top_k_indices = torch.log_softmax(initial_decoder_output, dim=0). \
            topk(beam_width, dim=0)

        # initialize beam
        for i in range(beam_width):
            log_p = top_k_log_probabilities[i].item()
            token_id = top_k_indices[i].item()
            current_beams[i].log_probabilities.append(log_p)
            current_beams[i].token_ids.append(token_id)
            current_beams[i].eos = token_id == eos_token_id

        search_depth += 1

        while beam_queue.qsize() < beam_width and search_depth < max_output_length:
            # build target input and target mask
            # run all beams that are not eos in parallel
            target_input_ids = torch.stack([b.get_decoder_input(device=device) for b in current_beams], dim=0)

            with torch.no_grad():
                repeated_encoder_outputs_i = encoder_outputs_i.repeat(1, target_input_ids.shape[0], 1) \
                    if encoder_outputs_i is not None else None
                decoder_output = model.decode(target_input_ids.T,
                                              repeated_encoder_outputs_i)[0][-1, :, :]

            log_softmax_scores = torch.log_softmax(decoder_output, dim=1).tolist()

            beam_candidates = []
            for beam_idx in range(len(log_softmax_scores)):
                for token_id, lp in enumerate(log_softmax_scores):
                    beam = current_beams[beam_idx]
                    entry = Beam(log_probabilities=beam.log_probabilities + [lp],
                                 token_ids=beam.token_ids + [token_id],
                                 eos=token_id == eos_token_id)
                    beam_candidates.append((entry, -score_fn(beam, input_strings[beam_idx])))

            # sort the candidates, and keep best num_candidates = 2 * beam_width (because then we have at least
            # beam_width non-eos beams in the candidates, this ensures we always have beam_width candidates
            # for each depth)
            sorted_beams = sorted(beam_candidates, key=lambda b: b[1])[:num_candidates]

            current_beams = []
            for beam, score in sorted_beams:
                if beam.eos:
                    beam_queue.put(
                        (score, beam)
                    )
                else:
                    current_beams.append(beam)

                if len(current_beams) >= beam_width:
                    break

            search_depth += 1

        # if there were not enough beams with eos found before reaching max_length, add the current beams to the queue
        if beam_queue.qsize() < beam_width:
            for beam in current_beams:
                beam_queue.put(
                    (-score_fn(beam, input_strings[b] if input_strings is not None else None), beam)
                )

        output_beams = [beam_queue.get()[1] for _ in range(beam_width)]
        all_beams.append(output_beams)

    return [
        [
            SequenceGenerationInferenceResult(
                token_ids=beam.token_ids,
                token_log_probabilities=beam.log_probabilities
            )
            for beam in batch_beams
        ]
        for batch_beams in all_beams
    ]
