import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import pprint
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List

from trt.model import tokenizer as toklib
from trt.utils import common, io

logger = common.get_logger("TOKENIZERS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        type=str,
                        choices=["bpe", "word_piece", "word_vocab", "char_vocab"],
                        help="type of the tokenizer to setup")
    parser.add_argument("--files",
                        type=str,
                        nargs="+",
                        required=True,
                        help="paths to files for tokenizer training")
    parser.add_argument("--vocab-size",
                        type=int,
                        required=True,
                        help="")
    parser.add_argument("--save",
                        type=str,
                        required=True,
                        help="path to save the tokenizer or vocab at")
    return parser.parse_args()


def _get_word_frequencies(file: str) -> Counter:
    frequencies = Counter()
    words = re.compile(r'(\w+|[^\w\s]+)')
    with open(file, "r", encoding="utf8") as f:
        for line in f:
            s = json.loads(line)["sequence"]
            for word in re.split(words, s):
                frequencies[word] += 1
    # save counts to temporary file to save memory
    tmp_file = str(hashlib.sha256(file.encode("utf8")).hexdigest()) + "_word_counts.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(frequencies, f)
    return tmp_file


def _get_char_frequencies(file: str) -> Counter:
    frequencies = Counter()
    with open(file, "r", encoding="utf8") as f:
        for line in f:
            s = json.loads(line)["sequence"]
            for char in s:
                frequencies[char] += 1
    # save counts to temporary file to save memory
    tmp_file = str(hashlib.sha256(file.encode("utf8")).hexdigest()) + "_char_counts.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(frequencies, f)
    return tmp_file


def combine_frequencies_from_files(files: List[str]) -> Dict[str, int]:
    logger.info("Combining frequencies from files")
    start = time.monotonic()
    counts = defaultdict(int)
    for tmp_file in files:
        with open(tmp_file, "rb") as f:
            count: Counter = pickle.load(f)
            for item, c in count.items():
                counts[item] += c
        os.remove(tmp_file)
    end = time.monotonic()
    logger.info(f"Combining frequencies took {end - start:.2f} seconds")
    return counts


def get_char_vocab_from_files(files: List[str]) -> Dict[str, int]:
    start = time.monotonic()
    with mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8)))) as pool:
        logger.info("Getting character frequencies from files")
        tmp_files = pool.map(_get_char_frequencies, files)
    end = time.monotonic()
    logger.info(f"Getting character frequencies took {end - start:.2f} seconds")
    return combine_frequencies_from_files(tmp_files)


def get_word_vocab_from_files(files: List[str]) -> Dict[str, int]:
    start = time.monotonic()
    with mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8)))) as pool:
        logger.info("Getting word frequencies from files")
        tmp_files = pool.map(_get_word_frequencies, files)
    end = time.monotonic()
    logger.info(f"Getting word frequencies took {end - start:.2f} seconds")
    return combine_frequencies_from_files(tmp_files)


def _get_files_from_globs(globs: str) -> List[str]:
    all_files = []
    for pattern in globs.split(","):
        pattern = pattern.strip()
        all_files.extend(io.glob_safe(pattern))
    return all_files


if __name__ == "__main__":
    args = parse_args()

    files = args.files
    logger.info(f"Training tokenizer/vocab on the following files: \n{pprint.pformat(files)}")

    if args.type == "word_vocab":
        word_vocab = get_word_vocab_from_files(files)
        with open(args.save, "w", encoding="utf8") as f:
            json.dump(word_vocab, f)
        tokenizer = toklib.get_word_vocab_tokenizer(vocab_size=args.vocab_size, vocab_path=args.save)

    elif args.type == "char_vocab":
        char_vocab = get_char_vocab_from_files(files)
        with open(args.save, "w", encoding="utf8") as f:
            json.dump(char_vocab, f)
        tokenizer = toklib.get_character_vocab_tokenizer(vocab_size=args.vocab_size, vocab_path=args.save)

    elif args.type == "bpe":
        tokenizer, bpe_trainer = toklib.get_bpe_tokenizer(vocab_size=args.vocab_size)
        tokenizer.train(bpe_trainer, files)
        toklib.save_tokenizer(tokenizer, args.save)

    else:
        tokenizer, word_piece_trainer = toklib.get_word_piece_tokenizer(vocab_size=args.vocab_size)
        tokenizer.train(word_piece_trainer, files)
        toklib.save_tokenizer(tokenizer, args.save)

    sentence = "This is a sentence to test the tokenization."
    logger.info(f"Testing tokenization with {args.type} tokenizer:\n"
                f"{sentence}\n"
                f"\u2192 {tokenizer.encode(sentence, pair=None).tokens}\n"
                f"\u2192 {tokenizer.decode(tokenizer.encode(sentence, pair=None).ids)}")
