import json
import string
from typing import Any, List, Tuple

import tokenizers
from tokenizers import PreTokenizedString, decoders, models, normalizers, pre_tokenizers, processors, trainers

from whitespace_correction.utils import constants
from whitespace_correction.utils.whitespace_correction import WhitespaceCorrectionTokens, get_correction_tokens


def _get_default_normalizer() -> normalizers.Sequence:
    return normalizers.Sequence([
        normalizers.Strip(),
        normalizers.StripAccents()
    ])


def _get_default_template_postprocessor(tokenizer: tokenizers.Tokenizer,
                                        template: str = f"{constants.BOS} $A {constants.EOS}",
                                        pair_template: str = f"{constants.BOS} $A {constants.SEP} $B {constants.EOS}",
                                        special_tokens: List[str] = [constants.BOS, constants.EOS, constants.SEP]) -> \
        processors.TemplateProcessing:
    return processors.TemplateProcessing(
        single=template,
        pair=pair_template,
        special_tokens=[(tok, tokenizer.token_to_id(tok)) for tok in special_tokens]
    )


def get_bpe_tokenizer(vocab_size: int) -> Tuple[tokenizers.Tokenizer, trainers.Trainer]:
    tokenizer = tokenizers.Tokenizer(models.BPE(unk_token=constants.UNK))
    tokenizer.normalizer = _get_default_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer, trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=constants.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )


def get_byte_tokenizer() -> tokenizers.Tokenizer:
    # sorted needed for determinism
    byte_alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {}
    token_id = 0
    for special_token in constants.SPECIAL_TOKENS:
        vocab[special_token] = token_id
        token_id += 1
    for b in byte_alphabet:
        vocab[b] = token_id
        token_id += 1
    tokenizer = tokenizers.Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=constants.UNK))
    tokenizer.normalizer = _get_default_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer


def get_word_piece_tokenizer(vocab_size: int) -> Tuple[tokenizers.Tokenizer, trainers.Trainer]:
    word_piece_prefix = "##"
    tokenizer = tokenizers.Tokenizer(models.WordPiece(unk_token=constants.UNK))
    tokenizer.normalizer = _get_default_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=word_piece_prefix)
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer, trainers.WordPieceTrainer(vocab_size=vocab_size,
                                                min_frequency=2,
                                                show_progress=True,
                                                special_tokens=constants.SPECIAL_TOKENS,
                                                limit_alphabet=1000,
                                                continuing_subword_prefix=word_piece_prefix)


def get_word_vocab_tokenizer(vocab_size: int, vocab_path: str) -> tokenizers.Tokenizer:
    with open(vocab_path, "r", encoding="utf8") as f:
        word_frequencies = json.load(f)
    vocab = {}
    token_id = 0
    for special_token in constants.SPECIAL_TOKENS:
        vocab[special_token] = token_id
        token_id += 1
    for word, _ in sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True):
        if token_id >= vocab_size:
            break
        vocab[word] = token_id
        token_id += 1
    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token=constants.UNK))
    tokenizer.normalizer = _get_default_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer


class _CharPreTokenizer:
    def _split(self, _: Any, normalized: tokenizers.NormalizedString) -> List[tokenizers.NormalizedString]:
        chars = []
        for char in str(normalized.normalized):
            chars.append(tokenizers.NormalizedString(char))
        return chars

    def pre_tokenize(self, pretokenized: PreTokenizedString) -> None:
        pretokenized.split(self._split)


class _JoinDecoder:
    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)


_ALL_CHARS = string.ascii_letters + string.digits + string.punctuation + " "


def get_character_tokenizer() -> tokenizers.Tokenizer:
    # can use word model here because we pre tokenize into chars (see _CharPreTokenizer() below)
    vocab = {}
    token_id = 0
    for special_token in constants.SPECIAL_TOKENS:
        vocab[special_token] = token_id
        token_id += 1
    # sorted needed for determinism
    for char in sorted(_ALL_CHARS):
        vocab[char] = token_id
        token_id += 1
    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token=constants.UNK))
    # tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_CharPreTokenizer())
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.decoder = decoders.Decoder.custom(_JoinDecoder())
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer


def get_character_vocab_tokenizer(vocab_size: int,
                                  vocab_path: str) -> tokenizers.Tokenizer:
    with open(vocab_path, "r", encoding="utf8") as f:
        char_frequencies = json.load(f)
    vocab = {}
    token_id = 0
    for special_token in constants.SPECIAL_TOKENS:
        vocab[special_token] = token_id
        token_id += 1
    for char, _ in sorted(char_frequencies.items(), key=lambda item: item[1], reverse=True):
        if token_id >= vocab_size:
            break
        vocab[char] = token_id
        token_id += 1
    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token=constants.UNK))
    tokenizer.normalizer = _get_default_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_CharPreTokenizer())
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.decoder = decoders.Decoder.custom(_JoinDecoder())
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer


class _WhitespaceCorrectionPreTokenizer:
    def _split(self, _: Any, normalized: tokenizers.NormalizedString) -> List[tokenizers.NormalizedString]:
        return [tokenizers.NormalizedString(token)
                for token in get_correction_tokens(str(normalized.normalized))]

    def pre_tokenize(self, pretokenized: PreTokenizedString) -> None:
        pretokenized.split(self._split)


def get_whitespace_correction_tokenizer() -> tokenizers.Tokenizer:
    vocab = {}
    token_id = 0
    for token in WhitespaceCorrectionTokens:
        vocab[token.value] = token_id
        token_id += 1
    for special_token in constants.SPECIAL_TOKENS:
        vocab[special_token] = token_id
        token_id += 1
    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token=constants.UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_WhitespaceCorrectionPreTokenizer())
    tokenizer.add_special_tokens(constants.SPECIAL_TOKENS)
    tokenizer.decoder = decoders.Decoder.custom(_JoinDecoder())
    tokenizer.post_processor = _get_default_template_postprocessor(tokenizer)
    return tokenizer


def save_tokenizer(tokenizer: tokenizers.Tokenizer, filepath: str) -> None:
    if not filepath.endswith(".json"):
        filepath += ".json"
    tokenizer.save(filepath, pretty=True)


def load_tokenizer(filepath: str, **kwargs: Any) -> tokenizers.Tokenizer:
    if filepath == "char":
        tokenizer = get_character_tokenizer()
    elif filepath == "char_vocab":
        tokenizer = get_character_vocab_tokenizer(vocab_size=kwargs["vocab_size"],
                                                  vocab_path=kwargs["vocab_path"])
    elif filepath == "word":
        tokenizer = get_word_vocab_tokenizer(vocab_size=kwargs["vocab_size"],
                                             vocab_path=kwargs["vocab_path"])
    elif filepath == "byte":
        tokenizer = get_byte_tokenizer()
    elif filepath == "whitespace_correction":
        tokenizer = get_whitespace_correction_tokenizer()
    else:
        tokenizer = tokenizers.Tokenizer.from_file(filepath)
    return tokenizer
