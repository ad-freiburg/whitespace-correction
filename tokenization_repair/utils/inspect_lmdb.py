import argparse
import pprint

import lmdb

import msgpack

from tqdm.auto import tqdm

from trt.model import tokenizer
from trt.utils import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("LMDB_INSPECTION")

    env = lmdb.open(args.lmdb,
                    map_size=10e11,  # approx. 100 GB
                    subdir=False,
                    max_readers=1,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)

    with env.begin(write=False) as txn:
        length = msgpack.loads(txn.get(b"__len__"))
        files = msgpack.loads(txn.get(b"__files__"))

        logger.info(f"LMDB stores {length} samples from {len(files)} files")
        logger.info(f"Files (first 50): \n{pprint.pformat(files[:50])}")

        all_lengths = []

        lengths_keys = msgpack.loads(txn.get(b"__lengths__"))
        for length_key in tqdm(lengths_keys, "Reading lengths"):
            lengths = msgpack.loads(txn.get(length_key))
            all_lengths.extend(lengths)

        all_lengths_sorted = sorted(all_lengths)
        num_tokens = sum(all_lengths_sorted)
        logger.info(f"Number of tokens in LMDB is {num_tokens}")
        logger.info(f"Average sample length is {num_tokens / len(all_lengths_sorted): .6f}")
        logger.info(f"Median sample length is {all_lengths_sorted[len(all_lengths_sorted) // 2]}")

        keys_keys = msgpack.loads(txn.get(b"__keys__"))
        idx = 0

        tok = tokenizer.load_tokenizer("char")

        failing_samples = []

        for keys_key in tqdm(keys_keys, "Reading samples"):
            keys = msgpack.loads(txn.get(keys_key))

            for key in keys:
                sample = msgpack.loads(txn.get(key))
                if not len(sample["input_ids"]) == all_lengths[idx] == len(sample["labels"]):
                    failing_samples.append((sample, all_lengths[idx]))
                    lens = {k: len(v) for k, v in sample.items()}
                    logger.info(
                        f"Found invalid sample at index {idx} with lengths {lens} that should be {all_lengths[idx]}: "
                        f"\n {sample}")
                    logger.info(tok.decode(sample["input_ids"]))
                    logger.info(tok.decode(sample["input_ids"], skip_special_tokens=False))

                idx += 1

        logger.info(f"Found {len(failing_samples)} invalid samples")
