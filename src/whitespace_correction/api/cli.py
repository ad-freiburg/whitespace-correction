from typing import Any, Iterator

from text_correction_utils.api.cli import TextCorrectionCli

from whitespace_correction import version
from whitespace_correction.api.corrector import WhitespaceCorrector
from whitespace_correction.api.server import WhitespaceCorrectionServer


class WhitespaceCorrectionCli(TextCorrectionCli):
    text_corrector_cls = WhitespaceCorrector
    text_correction_server_cls = WhitespaceCorrectionServer

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


def main():
    parser = WhitespaceCorrectionCli.parser(
        "Whitespace correction",
        "Correct missing or spurious whitespaces in text"
    )
    WhitespaceCorrectionCli(parser.parse_args()).run()
