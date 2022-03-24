import csv


def read_file(path: str):
    with open(path) as file:
        rows = [row for row in csv.reader(file)]
    return rows
