import argparse
from typing import Any, Iterator

from text_correction_utils.api import TextCorrectionCli

from whitespace_correction import version


class WhitespaceCorrectionCli(TextCorrectionCli):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def version(self) -> str:
        return version.__version__

    def parse_input(self, ipt: str) -> Any:
        return ipt.strip()

    def format_output(self, pred: Any) -> str:
        assert isinstance(pred, str), "only string outputs supported for whitespace correction"
        return pred

    def correct_iter(self, iter: Iterator[str]) -> Any:
        pass

    def correct_file(self, s: str) -> Any:
        pass
