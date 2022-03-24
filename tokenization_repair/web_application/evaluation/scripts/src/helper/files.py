from typing import List

import os
import shutil
import random

from src.settings.constants import FILE_ENCODING


def path_exists(path):
    return os.path.exists(path)


def file_exists(path):
    return os.path.isfile(path)


def make_directory(path):
    if not path_exists(path):
        os.mkdir(path)


def make_directory_recursive(path):
    parts = path.split("/")
    for i in range(2, len(parts)):
        subpath = "/".join(parts[:i])
        if not path_exists(subpath):
            make_directory(subpath)


def read_file(path):
    """
    Reads the content of a file as text.
    :param path: file
    :return: text
    """
    with open(path, "rb") as f:
        content = f.read().decode(FILE_ENCODING)
    return content


def read_lines(path):
    with open(path, encoding=FILE_ENCODING) as f:
        lines = [line[:-1] if len(line) > 0 and line[-1] == '\n' else line for line in f.readlines()]
    return lines


def write_file(path, content):
    path = path.encode(encoding="ascii", errors="replace")
    try:
        f = open(path, 'wb')
        f.write(content.encode(FILE_ENCODING))
        f.close()
    except UnicodeDecodeError:
        print(path)
        raise Exception("bla bli")


def write_lines(path: str, lines: List[str]):
    with open(path, 'w', encoding=FILE_ENCODING) as f:
        for line in lines:
            f.write(line + '\n')


def get_files(dir):
    return os.listdir(dir)


def parent_directory(dir):
    if dir[-1] == '/':
        dir = dir[:-1]
    split_path = dir.split('/')
    split_path[-1] = ''
    return '/'.join(split_path)


def remove_file(path):
    os.remove(path)


def remove_dir(dir):
    shutil.rmtree(dir)


def read_sequences(path):
    with open(path, encoding=FILE_ENCODING) as file:
        while True:
            line = file.readline()
            if line == "":
                break
            yield line[:-1]


def random_sequence_subset(path, n, seed):
    sequences = list(read_sequences(path))
    random.Random(seed).shuffle(sequences)
    return sequences[:n]


def open_file(path):
    return open(path, "w", encoding=FILE_ENCODING)


def copy_file(source, target):
    content = read_file(source)
    write_file(target, content)
