from io import TextIOWrapper
from typing import Iterator, Optional, Union

from text_utils.api.cli import TextProcessingCli
from text_utils import data

from whitespace_correction import version
from whitespace_correction.api.corrector import WhitespaceCorrector
from whitespace_correction.api.server import WhitespaceCorrectionServer


class WhitespaceCorrectionCli(TextProcessingCli):
    text_processor_cls = WhitespaceCorrector
    text_processing_server_cls = WhitespaceCorrectionServer

    def version(self) -> str:
        return version.__version__

    def process_iter(
        self,
        processor: WhitespaceCorrector,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from processor.correct_iter(
            ((data.text, data.language) for data in iter),
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            return_raw=True,
            show_progress=self.args.progress
        )

    def process_file(
        self,
        processor: WhitespaceCorrector,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        processor.correct_file(
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
