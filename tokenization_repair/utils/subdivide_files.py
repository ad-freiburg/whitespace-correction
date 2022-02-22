import argparse
import os
from typing import IO

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--lines-per-file", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def get_sub_file(file: str, out_dir: str, sub_file_idx: int, lines_per_file: int) -> IO:
    file_name, ext = os.path.splitext(os.path.basename(file))
    sub_file_name = f"{file_name}_{sub_file_idx}_{lines_per_file}{ext}"
    sub_file = open(os.path.join(out_dir, sub_file_name), "w", encoding="utf8")
    return sub_file


def subdivide(args: argparse.Namespace) -> None:
    if os.path.exists(args.out_dir):
        print(f"out directory {args.out_dir} already exists")
        return

    os.makedirs(args.out_dir)

    for file in tqdm(args.files, "Splitting files"):
        sub_file_idx = 0
        sub_file = get_sub_file(file, args.out_dir, sub_file_idx, args.lines_per_file)

        with open(file, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                sub_file.write(line.strip() + "\n")

                if (i + 1) % args.lines_per_file == 0:
                    sub_file.close()
                    sub_file_idx += 1
                    sub_file = get_sub_file(file, args.out_dir, sub_file_idx, args.lines_per_file)

        if sub_file:
            sub_file.close()


if __name__ == "__main__":
    subdivide(parse_args())
