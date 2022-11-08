import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ocr-error-files", type=str, required=True, nargs="+")
    parser.add_argument("--typo-files", type=str, required=True, nargs="+")
    return parser.parse_args()


def noise(args: argparse.Namespace) -> None:
    pass


if __name__ == "__main__":
    noise(parse_args())
