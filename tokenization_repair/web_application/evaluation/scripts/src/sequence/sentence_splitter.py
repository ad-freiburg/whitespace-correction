from typing import List

from nltk.tokenize import PunktSentenceTokenizer
from nltk import load

from src.helper.pickle import load_object
from src.settings import paths


def load_default_nltk_tokenizer() -> PunktSentenceTokenizer:
    return load('tokenizers/punkt/{0}.pickle'.format("english"))


class NLTKSentenceSplitter:
    def __init__(self):
        self.tokenizer = load_default_nltk_tokenizer()

    def split(self, text: str) -> List[str]:
        """Splits a text into sentences using the NLTK default Punkt tokenizer."""
        return self.tokenizer.tokenize(text)


class WikiPunktTokenizer:
    def __init__(self, trained_abbreviations: bool = False, extended_abbreviations: bool = True):
        """
        The default Punkt tokenizer with additional abbreviations.

        :param trained_abbreviations: Use the abbreviations of the Punkt tokenizer from Wikipedia.
        :param extended_abbreviations: Use the abbreviations determined by counting frequencies of tokens with and
        without dot on Wikipedia.
        """
        self.tokenizer = load_default_nltk_tokenizer()
        if trained_abbreviations:
            wiki_tokenizer = self._load_trained_tokenizer()
            for abbr in wiki_tokenizer._params.abbrev_types:
                self.tokenizer._params.abbrev_types.add(abbr)
        if extended_abbreviations:
            for abbr in load_object(paths.EXTENDED_PUNKT_ABBREVIATIONS):
                self.tokenizer._params.abbrev_types.add(abbr.lower())

    @staticmethod
    def _load_trained_tokenizer() -> PunktSentenceTokenizer:
        """Returns the Punkt tokenizer trained on Wikipedia."""
        return load_object(paths.WIKI_PUNKT_TOKENIZER)

    def _postprocess(self, text: str, sentences: str) -> List[str]:
        """Only allow splits at spaces."""
        postprocessed = [sentences[0]]
        text_pos = len(sentences[0])
        for sentence in sentences[1:]:
            if text_pos != ' ' and sentence == text[text_pos:(text_pos + len(sentence))]:
                postprocessed[-1] += sentence
                text_pos -= 1
            else:
                postprocessed.append(sentence)
            text_pos += len(sentence) + 1
        return postprocessed

    def split(self, text: str) -> List[str]:
        """Split a text into sentences."""
        sentences = self.tokenizer.tokenize(text)
        sentences = self._postprocess(text, sentences)
        return sentences
