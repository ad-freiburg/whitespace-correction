import argparse
import json
import multiprocessing as mp
import os
from typing import Tuple

import ftfy
from tqdm import tqdm

from whitespace_repair.utils import io, nlp


def txt_file_to_jsonl(in_and_out_file: Tuple[str, str]) -> None:
    in_file, out_file = in_and_out_file
    with open(in_file, "r", encoding="utf8") as txt_file, \
            open(out_file, "w", encoding="utf8") as json_file:
        for line in txt_file:
            line = ftfy.fix_text(nlp.clean_sequence(line.strip()))
            json.dump({"sequence": line}, json_file, ensure_ascii=False)
            json_file.write("\n")


def txt_to_jsonl(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    in_files = io.glob_safe(os.path.join(args.in_dir, "*.txt"))

    tasks = []
    for in_file in in_files:
        in_file_name = os.path.splitext(os.path.basename(in_file))[0]
        out_file = os.path.join(args.out_dir, f"{in_file_name}.jsonl")
        tasks.append((in_file, out_file))

    num_processes = int(os.environ.get("NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8)))

    with mp.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(txt_file_to_jsonl, tasks), total=len(tasks)):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    txt_to_jsonl(parse_args())
