import argparse
import pprint
from collections import Counter

import lmdb
import msgpack
from tqdm import tqdm

from whitespace_correction.utils import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=str, required=True)
    parser.add_argument("--class-distribution", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("LMDB_INSPECTION")

    env = lmdb.open(args.lmdb,
                    map_size=int(10e11),  # approx. 100 GB
                    subdir=False,
                    max_readers=1,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)

    classes = Counter()

    with env.begin(write=False) as txn:
        length = msgpack.loads(txn.get(b"__len__"))
        files = msgpack.loads(txn.get(b"__files__"))

        logger.info(f"LMDB stores {length} samples from {len(files)} files")
        logger.info(f"Files (first 20): \n{pprint.pformat(files[:20])}")

        all_lengths = []

        lengths_keys = msgpack.loads(txn.get(b"__lengths__"))
        for length_key in tqdm(lengths_keys, "Reading lengths"):
            lengths = msgpack.loads(txn.get(length_key))
            all_lengths.extend(lengths)

        all_lengths_sorted = sorted(all_lengths)
        num_tokens = sum(all_lengths_sorted)
        logger.info(f"Number of tokens in LMDB is {num_tokens:,}")
        logger.info(f"Average sample length is {num_tokens / len(all_lengths_sorted): .2f}")
        logger.info(f"Median sample length is {all_lengths_sorted[len(all_lengths_sorted) // 2]}")

        if args.class_distribution:
            key_keys = msgpack.loads(txn.get(b"__keys__"))
            for key_key in key_keys:
                for key in msgpack.loads(txn.get(key_key)):
                    data = msgpack.loads(txn.get(key))
                    for c in data["labels"]:
                        if c < 0:
                            continue
                        classes[c] += 1
            total_classes = sum(classes.values())
            relative_classes = {c: round(freq / total_classes, 2) for c, freq in classes.items()}
            logger.info(f"Class distribution in LMDB is\nabsolute:\n{dict(classes)}\nrelative:\n{relative_classes}")
