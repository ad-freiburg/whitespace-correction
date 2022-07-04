import argparse
import json
import multiprocessing as mp
import os
import random
import time
from typing import Dict, List, Optional, Union

import lmdb

import msgpack

import tokenizers

from whitespace_correction.model import tokenizer as toklib
from whitespace_correction.utils import common, data, io
from whitespace_correction.utils.config import DataPreprocessingConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to config file")
    return parser.parse_args()


logger = common.get_logger("DATA_PREPROCESSING")


def process_line(tokenizer: tokenizers.Tokenizer,
                 target_tokenizer: tokenizers.Tokenizer,
                 line: str,
                 pretokenize: bool,
                 ensure_equal_length: bool,
                 preprocessing_fn: Optional[data.PREPROCESSING_FN] = None) -> Optional[Dict[str, List[int]]]:
    json_obj: Dict[str, str] = json.loads(line)

    if preprocessing_fn is not None:
        preprocessed_json_obj = preprocessing_fn(json_obj)
        assert isinstance(preprocessed_json_obj, dict)
        json_obj.update(preprocessed_json_obj)

    sequence: Union[str, List[str]] = json_obj["sequence"]
    if pretokenize:
        sequence = tokenizer.pre_tokenizer.pre_tokenize_str(sequence)
        sequence = [item[0] for item in sequence]

    enc = tokenizer.encode(sequence, is_pretokenized=pretokenize, pair=None)
    enc_dict = {"input_ids": enc.ids}

    if "labels" in json_obj:
        enc_dict["labels"] = json_obj["labels"]
        if ensure_equal_length and len(enc_dict["labels"]) != len(enc_dict["input_ids"]):
            lengths = {k: len(v) for k, v in enc_dict.items()}
            logger.info(f"Skipping sample because lengths of input ids and labels are not equal: {lengths}\n"
                        f"{sequence} --> {enc_dict['labels']}")
            return None

    if "target_sequence" in json_obj:
        target_sequence: Union[str, List[str]] = json_obj["target_sequence"]
        if pretokenize:
            target_sequence = target_tokenizer.pre_tokenizer.pre_tokenize_str(target_sequence)
            target_sequence = [item[0] for item in target_sequence]

        enc = target_tokenizer.encode(target_sequence, is_pretokenized=pretokenize, pair=None)
        enc_dict["target_input_ids"] = enc.ids

        if ensure_equal_length and len(enc_dict["target_input_ids"]) != len(enc_dict["input_ids"]):
            lengths = {k: len(v) for k, v in enc_dict.items()}
            logger.info(f"Skipping sample because lengths of input ids and target ids are not equal: {lengths}\n"
                        f"{sequence} --> {target_sequence}")
            return None

    return enc_dict


def process_files(queue: mp.Queue,
                  files: List[str],
                  tokenizer_path: tokenizers.Tokenizer,
                  target_tokenizer_path: tokenizers.Tokenizer,
                  pretokenize: bool,
                  ensure_equal_length: bool,
                  preprocessing_fn: data.PREPROCESSING_FN,
                  max_sequence_length: int,
                  cut_overflowing: bool) -> None:
    tokenizer = toklib.load_tokenizer(tokenizer_path)
    target_tokenizer = toklib.load_tokenizer(target_tokenizer_path)
    for filepath in files:
        samples = []
        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                enc_dict = process_line(tokenizer,
                                        target_tokenizer,
                                        line,
                                        pretokenize=pretokenize,
                                        ensure_equal_length=ensure_equal_length,
                                        preprocessing_fn=preprocessing_fn)
                if enc_dict is None:
                    continue
                enc_length = max(
                    len(enc_dict["input_ids"]),
                    len(enc_dict.get("target_input_ids", [])),
                    len(enc_dict.get("labels", []))
                )
                if enc_length > max_sequence_length:
                    # if a sequence overflows we still can cut it instead of skipping it
                    # if the corresponding config is set
                    # should only be used when cutting off all sequences at some specific position is a sensible thing
                    # to do
                    if cut_overflowing:
                        enc_dict = {k: v[:max_sequence_length] for k, v in enc_dict.items()}
                        enc_length = max_sequence_length
                    else:
                        continue
                samples.append((enc_dict, enc_length))
        queue.put(samples)
    # signal to main process that this process is finished
    queue.put(None)


def write_lmdb(output_dir: str,
               lmdb_name: str,
               files: List[str],
               tokenizer_path: str,
               target_tokenizer_path: str,
               pretokenize: bool,
               ensure_equal_length: bool,
               preprocessing_fn: data.PREPROCESSING_FN,
               max_sequence_length: int,
               cut_overflowing: bool,
               max_sequences: int) -> None:
    env = lmdb.open(os.path.join(output_dir, lmdb_name),
                    subdir=False,
                    map_size=int(10e11),  # approx. 100 GB
                    readonly=False,
                    meminit=False,
                    map_async=True,
                    lock=False)
    start = time.monotonic()
    # overwrite / drop existing database
    db_handle = env.open_db()
    with env.begin(write=True) as txn:
        txn.drop(db_handle)

    # give each process a subset of the files
    queue: mp.Queue = mp.Queue()
    processes = []
    num_finished = 0
    num_processes = int(os.environ.get("NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8, len(files))))
    batch_size = len(files) // num_processes
    for i in range(num_processes):
        lower_idx = i * batch_size
        # last process gets all remaining files which could be more than batch size
        if i == (num_processes - 1):
            file_batch = files[lower_idx:]
        else:
            file_batch = files[lower_idx:lower_idx + batch_size]
        p = mp.Process(target=process_files,
                       args=(queue,
                             file_batch,
                             tokenizer_path,
                             target_tokenizer_path,
                             pretokenize,
                             ensure_equal_length,
                             preprocessing_fn,
                             max_sequence_length,
                             cut_overflowing))
        p.start()
        processes.append(p)
        logger.info(f"Started worker process {p.pid} on {len(file_batch)} files")

    lengths_keys = []
    lengths = []
    keys_keys = []
    keys = []
    num_sequences = 0

    txn = env.begin(write=True)

    txn.put(b"__files__", msgpack.dumps(files))

    while True:
        if num_sequences >= max_sequences:
            logger.info(f"Reached maximum sequences {max_sequences}")
            break
        if num_finished >= num_processes:
            logger.info(f"All processes are finished, processed {num_sequences} sequences")
            break

        samples = queue.get()
        if samples is None:
            num_finished += 1
            continue

        for enc_dict, enc_length in samples:
            key = f"{num_sequences}".encode("ascii")
            txn.put(key, msgpack.dumps(enc_dict))
            keys.append(key)
            lengths.append(enc_length)
            num_sequences += 1

            # commit every 1000000 samples if preprocessing is aborted
            if num_sequences % 1000000 == 0:
                _lengths_key = f"__lengths_upto_{num_sequences}__".encode("ascii")
                _keys_key = f"__keys_upto_{num_sequences}__".encode("ascii")

                txn.put(_keys_key, msgpack.dumps(keys))
                txn.put(_lengths_key, msgpack.dumps(lengths))

                keys_keys.append(_keys_key)
                lengths_keys.append(_lengths_key)

                txn.put(b"__len__", msgpack.dumps(num_sequences))
                txn.put(b"__keys__", msgpack.dumps(keys_keys))
                txn.put(b"__lengths__", msgpack.dumps(lengths_keys))

                txn.commit()
                txn = env.begin(write=True)

                lengths = []
                keys = []

            # log progress 100 times
            if num_sequences % max(max_sequences // 100, 1) == 0:
                end = time.monotonic()
                logger.info(
                    f"[{num_sequences}/{max_sequences}] Processed {num_sequences * 100 / max_sequences:.2f}% of"
                    f" all sequences, {common.eta_minutes((end - start) / 60, num_sequences, max_sequences)}")

            if num_sequences >= max_sequences:
                break

    for p in processes:
        logger.info(f"Stopping process {p.pid}")
        p.terminate()
        p.join()
        logger.info(f"Successfully stopped process {p.pid}")

    if len(keys) > 0 and len(lengths) > 0:
        _lengths_key = f"__lengths_upto_{num_sequences}__".encode("ascii")
        _keys_key = f"__keys_upto_{num_sequences}__".encode("ascii")

        txn.put(_keys_key, msgpack.dumps(keys))
        txn.put(_lengths_key, msgpack.dumps(lengths))

        keys_keys.append(_keys_key)
        lengths_keys.append(_lengths_key)

        txn.put(b"__len__", msgpack.dumps(num_sequences))
        txn.put(b"__keys__", msgpack.dumps(keys_keys))
        txn.put(b"__lengths__", msgpack.dumps(lengths_keys))

        txn.commit()


if __name__ == "__main__":
    args = parse_args()

    # disable parallelism for tokenizers explicitly
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    CONFIG = DataPreprocessingConfig.from_yaml(args.config)
    assert isinstance(CONFIG, DataPreprocessingConfig)

    logger.info(f"Using data preprocessing config:\n"
                f"{CONFIG}")

    os.makedirs(CONFIG.output_dir, exist_ok=True)

    # save copy of config file to output directory
    with open(os.path.join(CONFIG.output_dir, "config.yaml"), "w", encoding="utf8") as f:
        f.write(str(CONFIG))

    common.add_file_log(logger, os.path.join(CONFIG.output_dir, "logs.txt"))

    tokenizer = toklib.load_tokenizer(CONFIG.tokenizer)
    tokenizer_path = CONFIG.tokenizer

    if CONFIG.target_tokenizer is None:
        logger.info(f"No target tokenizer specified, reusing the tokenizer '{CONFIG.tokenizer}' "
                    f"for the target sequences if necessary")
        target_tokenizer = tokenizer
        target_tokenizer_path = tokenizer_path
    else:
        target_tokenizer = toklib.load_tokenizer(CONFIG.target_tokenizer)
        target_tokenizer_path = CONFIG.target_tokenizer

    test_sentence = "This is a sentence to test the preprocessing functions before the data preprocessing starts."
    logger.info(f"Testing tokenizer: {tokenizer.encode(test_sentence, pair=None).tokens}\n"
                f"Testing target tokenizer: {target_tokenizer.encode(test_sentence, pair=None).tokens}")

    if CONFIG.pretokenize:
        assert tokenizer.pre_tokenizer is not None and target_tokenizer.pre_tokenizer is not None, \
            "Expected that both the tokenizer and target tokenizer have pre tokenizers if pretokenize is set to true," \
            " but got None."
        logger.info("Pretokenize is set to True.\n"
                    f"Testing pre tokenizer: {tokenizer.pre_tokenizer.pre_tokenize_str(test_sentence)}\n"
                    f"Testing target pre tokenizer: {target_tokenizer.pre_tokenizer.pre_tokenize_str(test_sentence)}")

    if CONFIG.preprocessing is None:
        preprocessing_fn = None
    else:
        test_item = {"sequence": test_sentence, "target_sequence": test_sentence}
        corruption_fns = []
        for cfg in CONFIG.preprocessing:
            preprocessing_fn = data.get_preprocessing_fn(cfg.type, **cfg.arguments)
            logger.info(f"Testing '{cfg.type}' preprocessing function: {test_item} \u2192 "
                        f"{preprocessing_fn(test_item.copy())}")
            corruption_fns.append(preprocessing_fn)
        preprocessing_fn = data.chain_preprocessing_fns(corruption_fns)
        logger.info(f"Testing chained preprocessing function: {test_item} \u2192 {preprocessing_fn(test_item.copy())}")

    files = [file
             for g in CONFIG.data
             for file in io.glob_safe(g)]
    if CONFIG.seed is not None:
        rand = random.Random(CONFIG.seed)
        rand.shuffle(files)

    max_sequences = sum(io.line_count(file) for file in files)
    if CONFIG.max_sequences is not None:
        max_sequences = min(max_sequences, CONFIG.max_sequences)

    max_sequence_length = CONFIG.max_sequence_length if CONFIG.max_sequence_length is not None else float("inf")
    logger.info(f"Number of sequences limited to {max_sequences:,} "
                f"with a maximum sequence length of {max_sequence_length}")

    start = time.monotonic()
    write_lmdb(output_dir=CONFIG.output_dir,
               lmdb_name=CONFIG.lmdb_name,
               files=files,
               tokenizer_path=tokenizer_path,
               target_tokenizer_path=target_tokenizer_path,
               pretokenize=CONFIG.pretokenize,
               ensure_equal_length=CONFIG.ensure_equal_length,
               preprocessing_fn=preprocessing_fn,
               max_sequence_length=max_sequence_length,
               cut_overflowing=CONFIG.cut_overflowing,
               max_sequences=max_sequences)
    end = time.monotonic()

    logger.info(f"Finished preprocessing in {(end - start) / 60:.2f} minutes")
