import argparse
import os
import random

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def create_benchmark(args: argparse.Namespace) -> None:
    assert len(args.files) > 0, "got no input files"

    corrupt_lines = []
    correct_lines = []

    rand = random.Random(args.seed)

    for file in tqdm(args.files):
        with open(file, "r", encoding="utf8") as inf:
            for line in inf:
                words = line.strip().split()

                for i, (word, prev_word) in enumerate(zip(words[1:], words[:-1])):
                    if word[0] == prev_word[-1] and len(word) > 1 and len(prev_word) > 1:
                        # use up to 5 words of context
                        left_context = words[max(0, i - 5):max(0, i - 1)]
                        right_context = words[i + 2:i + 2 + 5]
                        corrupt_line = " ".join(
                            left_context + [prev_word[:-1]] + [word[0]] + [word[1:]] + right_context
                        )
                        r = rand.random()
                        if r < 0.1:
                            corrupt_lines.append(corrupt_line.replace(" ", ""))
                        elif r < 0.2:
                            corrupt_lines.append(" ".join(corrupt_line.replace(" ", "")))
                        else:
                            corrupt_lines.append(corrupt_line)
                        correct_lines.append([
                            " ".join(left_context + [prev_word] + [word[1:]] + right_context),
                            " ".join(left_context + [prev_word[:-1]] + [word] + right_context)
                        ])

    indices = list(range(len(corrupt_lines)))
    rand.shuffle(indices)
    indices = indices[:args.max_lines]

    with open(os.path.join(args.out_dir, "corrupt.txt"), "w", encoding="utf8") as corrupt_f, \
            open(os.path.join(args.out_dir, "correct.txt"), "w", encoding="utf8") as correct_f:
        for idx in indices:
            corrupt_f.write(corrupt_lines[idx] + "\n")
            correct_f.write("\t".join(correct_lines[idx]) + "\n")


if __name__ == "__main__":
    create_benchmark(parse_args())
