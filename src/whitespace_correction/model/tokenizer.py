import string
from typing import Any, List, Tuple, Dict

from whitespace_correction.utils import constants, whitespace_correction

Tokenization = Tuple[List[int], Dict[str, Any]]


class Tokenizer:
    vocab: Dict[str, int]

    def __init__(self, num_prefix_tokens: int = 1, num_suffix_tokens: int = 1):
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        self.eos_token_id = self.vocab[constants.EOS]
        self.bos_token_id = self.vocab[constants.BOS]
        self.unk_token_id = self.vocab[constants.UNK]
        self.pad_token_id = self.vocab[constants.PAD]

        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_added_tokens(self) -> int:
        return self.num_prefix_tokens + self.num_suffix_tokens

    def normalize(self, sequence: str) -> str:
        return self.normalize_batch([sequence])[0]

    def normalize_batch(self, sequences: List[str]) -> List[str]:
        return [whitespace_correction.clean_sequence(sequence) for sequence in sequences]

    def split(self, sequence: str) -> List[str]:
        return self.split_batch([sequence])[0]

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        raise NotImplementedError

    def tokenize(
            self,
            sequence: str,
            prefix_tokens: Tuple[str, ...] = (constants.BOS,),
            suffix_tokens: Tuple[str, ...] = (constants.EOS,)
    ) -> Tokenization:
        return self.tokenize_batch([sequence], prefix_tokens, suffix_tokens)[0]

    def tokenize_batch(
            self,
            sequences: List[str],
            prefix_tokens: Tuple[str, ...] = (constants.BOS,),
            suffix_tokens: Tuple[str, ...] = (constants.EOS,)
    ) -> List[Tokenization]:
        assert len(prefix_tokens) == self.num_prefix_tokens, \
            f"expected {self.num_prefix_tokens} prefix tokens, but got {len(prefix_tokens)}: {prefix_tokens}"
        assert len(suffix_tokens) == self.num_suffix_tokens, \
            f"expected {self.num_suffix_tokens} suffix tokens, but got {len(suffix_tokens)}: {suffix_tokens}"
        tokenizations = []
        for split in self.split_batch(sequences):
            token_ids = list(self.token_to_id(token) for token in prefix_tokens)
            for token in split:
                token_ids.append(self.token_to_id(token))
            token_ids.extend(self.token_to_id(token) for token in suffix_tokens)
            tokenizations.append((token_ids, {}))
        return tokenizations

    def de_tokenize(self, token_ids: List[int]) -> str:
        return self.de_tokenize_batch([token_ids])[0]

    def de_tokenize_batch(self, token_ids: List[List[int]]) -> List[str]:
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, constants.UNK)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


class ByteTokenizer(Tokenizer):
    def __init__(self, num_prefix_tokens: int = 1, num_suffix_tokens: int = 1) -> None:
        self.vocab = {
            **{chr(i): i for i in range(256)},
            **{st: 256 + i for i, st in enumerate(constants.SPECIAL_TOKENS)}
        }
        super().__init__(num_prefix_tokens, num_suffix_tokens)

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(chr(b) for b in sequence.encode("utf8")) for sequence in sequences]

    @staticmethod
    def _split_batch_extended(
            sequences: List[str],
            prefix_tokens: Tuple[str, ...] = (constants.BOS,),
            suffix_tokens: Tuple[str, ...] = (constants.EOS,)
    ) -> List[Tuple[List[str], List[int]]]:
        batch = []
        for sequence in sequences:
            splits = []
            char_groups = [1] * len(prefix_tokens)
            for char in sequence:
                char_bytes = char.encode("utf8")
                splits.extend(chr(b) for b in char_bytes)
                char_groups.append(len(char_bytes))
            char_groups.extend([1] * len(suffix_tokens))
            batch.append((splits, char_groups))
        return batch

    def tokenize_batch(
            self,
            sequences: List[str],
            prefix_tokens: Tuple[str, ...] = (constants.BOS,),
            suffix_tokens: Tuple[str, ...] = (constants.EOS,)
    ) -> List[Tokenization]:
        assert len(prefix_tokens) == self.num_prefix_tokens, \
            f"expected {self.num_prefix_tokens} prefix tokens, but got {len(prefix_tokens)}: {prefix_tokens}"
        assert len(suffix_tokens) == self.num_suffix_tokens, \
            f"expected {self.num_suffix_tokens} suffix tokens, but got {len(suffix_tokens)}: {suffix_tokens}"
        tokenizations = []
        for split, char_groups in self._split_batch_extended(sequences, prefix_tokens, suffix_tokens):
            token_ids = list(self.token_to_id(token) for token in prefix_tokens)
            for token in split:
                token_ids.append(self.token_to_id(token))
            token_ids.extend(self.token_to_id(token) for token in suffix_tokens)
            tokenizations.append((token_ids, {"char_groups": char_groups}))
        return tokenizations

    def de_tokenize_batch(self, token_ids: List[List[int]]) -> List[str]:
        sequences = []
        for ids in token_ids:
            sequences.append(bytes(ids).decode("utf8"))
        return sequences


_ALL_CHARS = string.ascii_letters + string.digits + string.punctuation + " "


class CharacterTokenizer(Tokenizer):
    def __init__(self, num_prefix_tokens: int = 1, num_suffix_tokens: int = 1) -> None:
        self.vocab = {
            **{c: i for i, c in enumerate(_ALL_CHARS)},
            **{st: len(_ALL_CHARS) + i for i, st in enumerate(constants.SPECIAL_TOKENS)}
        }
        super().__init__(num_prefix_tokens, num_suffix_tokens)

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(sequence) for sequence in sequences]

    def de_tokenize_batch(self, token_ids: List[List[int]]) -> List[str]:
        sequences = []
        for ids in token_ids:
            sequences.append("".join(self.id_to_token(token_id) for token_id in ids))
        return sequences


def load_tokenizer(name: str) -> Tokenizer:
    if name == "char":
        return CharacterTokenizer()
    elif name == "byte":
        return ByteTokenizer()
    else:
        raise ValueError(f"unknown tokenizer {name}")
