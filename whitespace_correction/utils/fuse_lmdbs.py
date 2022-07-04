import argparse
import os
import shutil
import time

import lmdb

import msgpack

from tqdm.auto import tqdm

from whitespace_correction.utils import common


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-lmdbs", type=str, nargs="+", required=True)
    parser.add_argument("--out-lmdb", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("FUSE_LMDBS")

    os.makedirs(os.path.dirname(args.out_lmdb), exist_ok=True)
    shutil.copy2(args.in_lmdbs[0], args.out_lmdb)

    logger.info(f"Copying {args.in_lmdbs[0]} into {args.out_lmdb} as base lmdb")

    out_env = lmdb.open(args.out_lmdb,
                        subdir=False,
                        map_size=10e11,
                        readonly=False,
                        meminit=False,
                        map_async=True,
                        lock=False)

    start = time.monotonic()

    with out_env.begin(write=False) as out_txn:
        out_length = msgpack.loads(out_txn.get(b"__len__"))
        out_lengths_keys = msgpack.loads(out_txn.get(b"__lengths__"))
        out_keys_keys = msgpack.loads(out_txn.get(b"__keys__"))
        out_files = set(msgpack.loads(out_txn.get(b"__files__")))

    for i in range(1, len(args.in_lmdbs)):
        in_lmdb = args.in_lmdbs[i]
        fuse_start = time.monotonic()

        logger.info(f"Fusing {in_lmdb}")

        in_env = lmdb.open(in_lmdb,
                           map_size=10e11,  # approx. 100 GB
                           subdir=False,
                           max_readers=1,
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False)

        with out_env.begin(write=True) as out_txn, in_env.begin(write=False) as in_txn:

            in_length = msgpack.loads(in_txn.get(b"__len__"))
            out_length += in_length

            in_files = msgpack.loads(in_txn.get(b"__files__"))
            out_files.union(set(in_files))

            in_lengths_keys = msgpack.loads(in_txn.get(b"__lengths__"))
            for in_length_key in tqdm(in_lengths_keys, "Fusing lengths"):
                in_lengths = msgpack.loads(in_txn.get(in_length_key))

                out_length_key = (in_length_key.decode("ascii") + f"_{i}").encode("ascii")

                out_txn.put(out_length_key, msgpack.dumps(in_lengths))
                out_lengths_keys.append(out_length_key)

            in_keys_keys = msgpack.loads(in_txn.get(b"__keys__"))
            in_keys = []
            for in_key_key in tqdm(in_keys_keys, "Fusing samples"):
                in_keys = msgpack.loads(in_txn.get(in_key_key))

                out_key_key = (in_key_key.decode("ascii") + f"_{i}").encode("ascii")

                out_keys_keys.append(out_key_key)

                new_out_keys = []

                for in_key in in_keys:
                    out_key = (in_key.decode("ascii") + f"_{i}").encode("ascii")

                    sample = msgpack.loads(in_txn.get(in_key))

                    out_txn.put(out_key, msgpack.dumps(sample), overwrite=False)

                    new_out_keys.append(out_key)

                out_txn.put(out_key_key, msgpack.dumps(new_out_keys))

        fuse_end = time.monotonic()
        logger.info(f"Fusing {in_length} samples from {in_lmdb} took {(fuse_end - fuse_start) / 60:.2f} minutes")

    with out_env.begin(write=True) as out_txn:
        out_txn.put(b"__len__", msgpack.dumps(out_length), overwrite=True)
        out_txn.put(b"__lengths__", msgpack.dumps(out_lengths_keys), overwrite=True)
        out_txn.put(b"__keys__", msgpack.dumps(out_keys_keys), overwrite=True)
        out_txn.put(b"__files__", msgpack.dumps(list(out_files)), overwrite=True)

        new_lengths = msgpack.loads(out_txn.get(out_lengths_keys[0]))
        new_keys = msgpack.loads(out_txn.get(out_keys_keys[0]))

    end = time.monotonic()

    logger.info(f"Finished fusing in {(end - start) / 60:.2f} minutes")
