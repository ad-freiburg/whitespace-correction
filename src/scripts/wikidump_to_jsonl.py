import argparse
import json
import re
import multiprocessing as mp
import os
from typing import Tuple, Optional
try:
    from spacy.language import Language
except ImportError:
    print("to use this script for preprocessing wikidump files, you must install spacy")
    exit()

from tqdm import tqdm

from whitespace_correction.utils import whitespace_correction

_DOC_REGEX = re.compile(r"<doc id=\"(\d+)\".*?>(.*?)</doc>", re.DOTALL)
_DOC_TAG_REGEX = re.compile(r"<doc id=\"\d+\".*?>|</doc>")
_CHAR_REGEX = re.compile(r"[a-zA-Z]+")
_MARKUP_REGEX = re.compile(r"<.*?>|</.*?>")


def invalid_sequence(sequence: str, min_length: int = 1) -> bool:
    return (
            len(sequence) < min_length
            or _CHAR_REGEX.search(sequence) is None
            or _MARKUP_REGEX.search(sequence) is not None
    )


def get_spacy_from_language_code(code: str) -> Language:
    from spacy.lang.en import English
    from spacy.lang.es import Spanish
    from spacy.lang.de import German
    from spacy.lang.pt import Portuguese
    from spacy.lang.it import Italian
    from spacy.lang.fr import French
    if code == "en":
        lang = English()
    elif code == "es":
        lang = Spanish()
    elif code == "de":
        lang = German()
    elif code == "pt":
        lang = Portuguese()
    elif code == "it":
        lang = Italian()
    elif code == "fr":
        lang = French()
    else:
        raise ValueError(f"unknown language code {code}")

    lang.max_length = 1e8
    lang.add_pipe("sentencizer")
    return lang


def wiki_file_to_jsonl(args: Tuple[str, str, Optional[str]]) -> None:
    in_file, out_file, lang_code = args

    with open(in_file, "r", encoding="utf8") as inf:
        text = inf.read()

    samples = []
    if lang_code is None:
        # split by paragraphs
        for line in text.splitlines():
            if _DOC_TAG_REGEX.match(line) is not None or invalid_sequence(line):
                continue
            line = whitespace_correction.clean_sequence(line)
            samples.append({"sequence": line})
    else:
        # split by sentences using the given language
        lang = get_spacy_from_language_code(lang_code)
        for match in re.finditer(_DOC_REGEX, text):
            g = whitespace_correction.clean_sequence(match.group(2))
            docs = lang.pipe([g])
            for doc in docs:
                for s in doc.sents:
                    sequence = str(s)
                    if invalid_sequence(sequence):
                        continue
                    samples.append({"sequence": sequence})

    with open(out_file, "w", encoding="utf8") as of:
        for sample in samples:
            json.dump(sample, of, ensure_ascii=False)
            of.write("\n")


def wiki_to_jsonl(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for in_file in args.in_files:
        in_file_name = os.path.splitext(os.path.basename(in_file))[0]
        out_file = os.path.join(args.out_dir, f"{in_file_name}.jsonl")
        tasks.append((in_file, out_file, args.language if args.split == "sentences" else None))

    num_processes = int(os.environ.get("NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8)))

    with mp.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(wiki_file_to_jsonl, tasks), total=len(tasks)):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-files", type=str, required=True, nargs="+")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", choices=["paragraphs", "sentences"], default="paragraphs")
    parser.add_argument("--language", choices=["en", "de", "es", "fr", "it", "pt"], default="en")
    return parser.parse_args()


if __name__ == "__main__":
    wiki_to_jsonl(parse_args())
