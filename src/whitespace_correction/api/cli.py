from io import TextIOWrapper
from typing import Iterator, Optional, Union

from text_correction_utils.api.cli import TextCorrectionCli
from text_correction_utils import data

from whitespace_correction import version
from whitespace_correction.api.corrector import WhitespaceCorrector
from whitespace_correction.api.server import WhitespaceCorrectionServer


class WhitespaceCorrectionCli(TextCorrectionCli):
    text_corrector_cls = WhitespaceCorrector
    text_correction_server_cls = WhitespaceCorrectionServer

    def version(self) -> str:
        return version.__version__

    def correct_iter(
        self,
        corrector: WhitespaceCorrector,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from corrector.correct_iter(
            ((data.text, data.language) for data in iter),
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            return_raw=True,
            show_progress=self.args.progress
        )

    def correct_file(
        self,
        corrector: WhitespaceCorrector,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        corrector.correct_file(
            path,
            self.args.input_format,
            out_file,
            self.args.output_format,
            lang,
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            show_progress=self.args.progress
        )


def main():
    import warnings
    warnings.filterwarnings("ignore")
    parser = WhitespaceCorrectionCli.parser(
        "Whitespace correction",
        "Correct missing or spurious whitespaces in text"
    )
    WhitespaceCorrectionCli(parser.parse_args()).run()
