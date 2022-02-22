import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from typing import List, Set, Tuple

from spacy.lang.en import English

from trt.utils import common, io
from trt.utils.data import SAMPLE
from trt.utils.nlp import clean_sequence, clean_text, is_valid_sequence, tokens_to_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Path to input directory")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("-d", "--dataset", type=str, choices=[
        "bea2019",
        "github_typo",
        "bookcorpus",
        "wiki",
        "tokenization_repair"
    ], required=True, help="Dataset to setup")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset setup")
    return parser.parse_args()


def _save_samples_to_jsonl(samples: List[SAMPLE],
                           save_to: str,
                           mode: str = "w",
                           verbose: bool = True):
    if verbose:
        logger.info(f"Saving {len(samples)} samples to {save_to}")
    d, _ = os.path.split(save_to)
    os.makedirs(d, exist_ok=True)
    with open(save_to, mode, encoding="utf8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _check_paths(filepath: str, save_to: str, overwrite: bool) -> bool:
    if not _exists(filepath):
        logger.info(f"Skipping processing of {filepath} since it does not exist.")
        return False
    if _exists(save_to) and not overwrite:
        logger.info(f"Skipping processing of {filepath} since {save_to} already exists. "
                    f"Set corresponding overwrite flag to force processing.")
        return False
    logger.info(f"Processing {filepath}...")
    return True


def process_m2_file(filepath: str, save_to: str, overwrite: bool = False):
    if not _check_paths(filepath, save_to, overwrite):
        return
    samples = _process_m2_file(filepath)
    _save_samples_to_jsonl(samples, save_to=save_to)


def process_m2_dir(directory: str, save_to: str, overwrite: bool = False):
    os.makedirs(save_to, exist_ok=True)
    files = io.glob_safe(os.path.join(directory, "*.m2"))
    for file in files:
        process_m2_file(file, save_to=os.path.join(save_to, os.path.split(file)[1] + ".jsonl"), overwrite=overwrite)


def _process_m2_file(filepath: str) -> List[SAMPLE]:
    with open(filepath, "r", encoding="utf8") as f:
        raw = f.read()

    m2_blocks_re = re.compile(r"(?:\r?\n){2,}")

    m2_blocks = re.split(m2_blocks_re, raw)
    samples = [sample for block in m2_blocks for sample in _parse_m2_block(block)]

    return samples


def _parse_m2_block(s: str) -> List[SAMPLE]:
    samples = []
    lines = s.splitlines()
    if len(lines) == 0:
        return samples
    assert lines[0].startswith("S")
    orig_tokens = lines[0].split(" ")[1:]
    annotations = defaultdict(list)
    for i in range(1, len(lines)):
        assert lines[i].startswith("A")
        corr = lines[i].split("|||")
        assert len(corr) == 6, f"expected corr to be of length 3, but got {corr}"
        prefix, error_type, correction, req, _, ann_id = corr
        _, from_idx, to_idx = prefix.split(" ")
        correction = correction.split(" ")
        if not isinstance(correction, list):
            correction = [correction]
        annotations[ann_id].append((int(from_idx), int(to_idx), correction, error_type))

    for ann_id, corrections in annotations.items():
        corrected_tokens = copy.deepcopy(orig_tokens)
        # account for changes of the indices due to insertions or deletions
        length_change = 0
        for from_idx, to_idx, correction, error_type in corrections:
            from_idx += length_change
            to_idx += length_change
            if error_type.strip() == "noop":
                continue
            del corrected_tokens[from_idx: to_idx]
            for corr in correction:
                corrected_tokens.insert(from_idx, corr)
                from_idx += 1
            length_change = len(corrected_tokens) - len(orig_tokens)
        item = {"sequence": clean_text(tokens_to_text(orig_tokens)),
                "target_sequence": clean_text(tokens_to_text(corrected_tokens))}
        samples.append(item)
    return samples


def process_github_typo(filepath: str, save_to: str, overwrite: bool = False):
    if not _check_paths(filepath, save_to, overwrite):
        return

    def _process_line(line: str) -> List[SAMPLE]:
        try:
            json_obj = json.loads(line, strict=False)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
        samples = []
        for edit in json_obj["edits"]:
            if "is_typo" not in edit:
                continue
            if not edit["is_typo"]:
                continue
            src = edit["src"]
            tgt = edit["tgt"]
            if src["lang"] != "eng" or tgt["lang"] != "eng":
                continue
            sample: SAMPLE = {"sequence": clean_text(str(src["text"])),
                              "target_sequence": clean_text(str(tgt["text"]))}
            samples.append(sample)
        return samples

    with open(filepath, "r", encoding="utf8") as f:
        data: List[SAMPLE] = [sample
                              for line in f
                              for sample in _process_line(line)]
    _save_samples_to_jsonl(samples=data,
                           save_to=save_to)


def _process_wikidump_file(fp: str, save_to: str, overwrite: bool, invalid_doc_ids: Set[int]):
    if not _check_paths(fp, save_to, overwrite):
        return
    doc_regex = re.compile(r"<doc id=\"(\d+)\".*?>(.*?)</doc>", re.DOTALL)
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.max_length = 1e7

    samples: List[SAMPLE] = []
    num_invalid = 0

    with open(fp, "r", encoding="utf8") as f:
        raw = f.read()

    for match in re.finditer(doc_regex, raw):
        doc_id = int(match.group(1))
        if doc_id in invalid_doc_ids:
            logger.info(f"Skipping document with doc id {doc_id}")
            continue

        g = match.group(2)
        docs = nlp.pipe([g])
        for doc in docs:
            for s in doc.sents:
                sample = clean_text(clean_sequence(str(s)))
                if not is_valid_sequence(sample, min_length=1):
                    num_invalid += 1
                    continue
                samples.append({"sequence": sample})
    logger.info(
        f"Percentage of invalid sequences in file {fp} is "
        f"{(num_invalid * 100) / max(1, num_invalid + len(samples)):.2f} %")
    _save_samples_to_jsonl(samples, save_to=save_to, verbose=False)


def process_wikidump(directory: str, save_to: str, overwrite: bool = False):
    files = sorted(io.glob_safe(os.path.join(directory, "extracted", "*", "wiki_*")))
    save_tos = [os.path.join(save_to, file.split("/")[-2], file.split("/")[-1] + ".jsonl") for file in files]
    overwrites = [overwrite] * len(files)

    invalid_doc_ids = set()
    with open(os.path.join(directory, "wikipedia_development_article_ids.txt"), "r", encoding="utf8") as dev_ids:
        for line in dev_ids:
            invalid_doc_ids.add(int(line.strip()))

    with open(os.path.join(directory, "wikipedia_test_article_ids.txt"), "r", encoding="utf8") as test_ids:
        for line in test_ids:
            invalid_doc_ids.add(int(line.strip()))

    logger.info(f"Will ignore {len(invalid_doc_ids)} documents because they are used for dev and test")

    invalid_doc_ids = [invalid_doc_ids] * len(files)

    start = time.monotonic()
    pool = mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8))))
    result = pool.starmap_async(_process_wikidump_file, list(zip(files, save_tos, overwrites, invalid_doc_ids)))
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Processed {finished}/{tasks} chunks\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()


def _process_bookcorpus_file(fp: str, save_to: str, overwrite: bool):
    if not _check_paths(fp, save_to, overwrite):
        return

    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.max_length = 1e7

    num_invalid = 0
    samples: List[SAMPLE] = []

    with open(fp, "r", encoding="utf8") as f:
        raw = f.read()

    docs = nlp.pipe([raw])
    for doc in docs:
        for s in doc.sents:
            sample = clean_text(clean_sequence(str(s)))
            if not is_valid_sequence(sample, min_length=1):
                num_invalid += 1
                continue
            samples.append({"sequence": sample})

    logger.info(
        f"Percentage of invalid sequences in file {fp} is "
        f"{(num_invalid * 100) / max(1, num_invalid + len(samples)):.2f} %")
    _save_samples_to_jsonl(samples, save_to=save_to, verbose=False)


def process_bookcorpus(directory: str, save_to: str, overwrite: bool = False):
    files = sorted(io.glob_safe(os.path.join(directory, "*.epub.txt")))
    save_tos = [os.path.join(save_to, os.path.split(file)[-1] + ".jsonl") for file in files]
    overwrites = [overwrite] * len(files)

    start = time.monotonic()
    pool = mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8))))
    result = pool.starmap_async(_process_bookcorpus_file, list(zip(files, save_tos, overwrites)))
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Processed {finished}/{tasks} chunks\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()


def find_matches_ignore_space(queries: List[str], text: str):
    """

    Search if string_1 is in string_2, but ignore potentially different white spacing

    :param queries: list of strings to search for in text
    :param text: some text
    :return:
    """

    text = re.sub(r"\s", " ", text).strip()
    assert text != "", "text cannot be empty"

    queries = [query.strip() for query in queries]

    text_no_spaces = ""

    num_spaces_until = [0]

    for i, c in enumerate(text):
        if c != " ":
            text_no_spaces += c
            if i < len(text) - 1:
                last_num = num_spaces_until[-1]
                num_spaces_until.append(last_num)
        else:
            num_spaces_until[-1] += 1

    assert len(num_spaces_until) == len(text_no_spaces), f"Got {len(num_spaces_until)} and {len(text_no_spaces)}"

    matches = []

    last_query_match_pos = 0

    for query in queries:
        query_no_spaces = ""
        query_space_positions = []
        for i, c in enumerate(query):
            if re.match(r"\S", c):
                query_no_spaces += c
            else:
                query_space_positions.append(i)

        assert query_no_spaces != "", "query cannot be empty"

        pos = text_no_spaces.find(query_no_spaces, last_query_match_pos)
        if pos == -1:
            matches.append(None)
            continue

        end_pos = pos + len(query_no_spaces)
        matches.append((pos + num_spaces_until[pos], end_pos + num_spaces_until[end_pos - 1]))
        last_query_match_pos = pos

    return [(queries[i], text[match[0]:match[1]]) for i, match in enumerate(matches) if match is not None]


def _process_arxiv_file(gt_filepath: str, extracted_filepath: str, save_to: str, overwrite: bool):
    if not _check_paths(gt_filepath, save_to, overwrite):
        return

    if not _check_paths(extracted_filepath, save_to, overwrite):
        return

    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.max_length = 1e7

    samples: List[SAMPLE] = []
    equal = 0

    with open(gt_filepath, "r", encoding="utf8") as gf, open(extracted_filepath, "r", encoding="utf8") as ef:
        raw_groundtruth = gf.readlines()
        raw_extracted = ef.read().strip()

        if raw_extracted == "":
            return

        queries = []
        for gt in raw_groundtruth:
            docs = nlp.pipe([gt.strip()])
            for doc in docs:
                for s in doc.sents:
                    seq = str(s).strip()
                    if seq == "" or "[formula]" in seq:
                        continue
                    queries.append(seq)

        matches = find_matches_ignore_space(queries, raw_extracted)

        for gt, ex in matches:
            samples.append({"sequence": ex, "target_sequence": gt})
            if ex == gt:
                equal += 1

    if len(samples) == 0:
        return

    logger.info(f"{(equal / len(samples)) * 100:.2f}% of samples are the same in {gt_filepath}")

    _save_samples_to_jsonl(samples, save_to=save_to, verbose=False)


def process_arxiv_dataset(directory: str, save_to: str, overwrite: bool = False):
    num_processes = int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8)))

    training_files_path = os.path.join(directory, "training_files.txt")
    with open(training_files_path, "r", encoding="utf8") as tf:
        training_files = tf.readlines()

    training_files = set([tf.strip() for tf in training_files])

    groundtruth_base_path = os.path.join("groundtruth", "body-text")
    extracted_base_path = os.path.join("extracted", "extraction-tool-results", "pdftotext")

    groundtruth_files = io.glob_safe(os.path.join(directory, groundtruth_base_path, "*/*.body.txt"))
    extracted_files = io.glob_safe(os.path.join(directory, extracted_base_path, "*/*.txt"))

    def _match_training(filepath: str) -> Tuple[str, bool]:
        split_path = filepath.split("/")
        if filepath.endswith(".body.txt"):
            new_filepath = os.path.join(split_path[-2], split_path[-1])
        else:
            base, ext = os.path.splitext(split_path[-1])
            new_filepath = os.path.join(split_path[-2], f"{base}.body{ext}")
        return new_filepath, new_filepath in training_files

    new_groundtruth_files = set()
    for gf in groundtruth_files:
        new_gf, is_in_training = _match_training(gf)
        if is_in_training:
            new_groundtruth_files.add(new_gf)

    new_extracted_files = set()
    for ef in extracted_files:
        new_ef, is_in_training = _match_training(ef)
        if is_in_training:
            new_extracted_files.add(new_ef)

    training_files = list(new_groundtruth_files.intersection(new_extracted_files))
    logger.info(f"Number of arXiv training files is {len(training_files)}")
    save_tos = [os.path.join(save_to, os.path.split(tf)[-1] + ".jsonl") for tf in training_files]
    gt_files = [os.path.join(directory, groundtruth_base_path, tf) for tf in training_files]
    ex_files = [os.path.join(directory, extracted_base_path, tf.replace(".body.txt", ".txt")) for tf in training_files]
    overwrites = [overwrite] * len(training_files)

    start = time.monotonic()
    pool = mp.Pool(num_processes)
    arguments = list(zip(gt_files, ex_files, save_tos, overwrites))
    result = pool.starmap_async(_process_arxiv_file, arguments)
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Processed {finished}/{tasks} files\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()


def _process_tokenization_repair_chunk(file_path: str, save_to: str):
    samples = []
    with open(file_path, "r") as inf:
        for line in inf:
            line = line.strip()
            sample = {"sequence": clean_sequence(clean_text(line))}
            samples.append(sample)

    _save_samples_to_jsonl(samples, save_to, verbose=False)


def process_tokenization_repair(directory: str, save_to: str, overwrite: bool = False):
    if not _check_paths(directory, save_to, overwrite):
        return

    num_processes = int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8)))

    start = time.monotonic()
    pool = mp.Pool(num_processes)

    files = io.glob_safe(os.path.join(directory, "*.txt"))
    save_tos = [os.path.join(save_to, f"{os.path.basename(file)}.jsonl") for file in files]

    arguments = list(zip(files, save_tos))
    result = pool.starmap_async(_process_tokenization_repair_chunk, arguments)
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Finished {finished}/{tasks} tasks\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("CLEAN_AND_FORMAT")

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    if args.dataset == "arxiv":
        ARXIV = os.path.join(INPUT_DIR, "arxiv_dataset")
        process_arxiv_dataset(ARXIV, os.path.join(OUTPUT_DIR, "arxiv"), overwrite=args.overwrite)

    elif args.dataset == "bookcorpus":
        BOOKCORPUS = os.path.join(INPUT_DIR, "bookcorpus/books1/epubtxt")
        process_bookcorpus(BOOKCORPUS, os.path.join(OUTPUT_DIR, "bookcorpus"), overwrite=args.overwrite)

    elif args.dataset == "bea2019":
        FCE_GLOB = os.path.join(INPUT_DIR, "bea/fce_v2.1.bea19/fce/m2")
        process_m2_dir(FCE_GLOB, os.path.join(OUTPUT_DIR, "fce"), overwrite=args.overwrite)

        LANG8 = os.path.join(INPUT_DIR, "bea/lang8.bea19/lang8.train.auto.bea19.m2")
        process_m2_file(LANG8, os.path.join(OUTPUT_DIR, "lang8.m2.jsonl"), overwrite=args.overwrite)

        WILOCNESS_GLOB = os.path.join(INPUT_DIR, "bea/wi+locness_v2.1.bea19/wi+locness/m2")
        process_m2_dir(WILOCNESS_GLOB, os.path.join(OUTPUT_DIR, "wilocness"), overwrite=args.overwrite)

        NUCLE = os.path.join(INPUT_DIR, "bea/nucle/nucle.train.gold.bea19.m2")
        process_m2_file(NUCLE, os.path.join(OUTPUT_DIR, "nucle.m2.jsonl"), overwrite=args.overwrite)

        GEC = os.path.join(INPUT_DIR, "nus/10gec_annotations")
        process_m2_dir(GEC, os.path.join(OUTPUT_DIR, "gec"), overwrite=args.overwrite)

    elif args.dataset == "github_typo":
        GITHUB_TYPO = os.path.join(INPUT_DIR, "github_typo/github-typo-corpus.v1.0.0.jsonl")
        process_github_typo(GITHUB_TYPO, os.path.join(OUTPUT_DIR, "github_typo.jsonl"),
                            overwrite=args.overwrite)

    elif args.dataset == "tokenization_repair":
        TOK_REP = os.path.join(INPUT_DIR, "tokenization_repair_mixed_split")
        process_tokenization_repair(TOK_REP, os.path.join(OUTPUT_DIR, "tokenization_repair", "mixed"),
                                    overwrite=args.overwrite)
        TOK_REP = os.path.join(INPUT_DIR, "tokenization_repair_mixed_ocr_spelling_errors_split")
        process_tokenization_repair(TOK_REP, os.path.join(OUTPUT_DIR, "tokenization_repair", "mixed_with_errors"),
                                    overwrite=args.overwrite)

    else:
        WIKIDUMP = os.path.join(INPUT_DIR, "wikidump_20201020")
        process_wikidump(WIKIDUMP, os.path.join(OUTPUT_DIR, "wikidump"), overwrite=args.overwrite)
