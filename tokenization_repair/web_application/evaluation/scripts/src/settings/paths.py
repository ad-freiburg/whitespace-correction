from typing import Optional

from src.helper.files import path_exists, make_directory
from src.benchmark.subset import Subset

# BASE DIRECTORY
DUMP_DIRS = [
    "data/",
    #"/home/hertel/tokenization-repair-dumps/data_naacl2021/",  # repro
    "/home/hertel/tokenization-repair-dumps/data/",  # wunderfitz
    "/local/data/hertelm/tokenization-repair-dumps/data/",  # sirba
    "/data/1/matthias-hertel/tokenization-repair-dumps/data/",  # polyaxon
    "/external/"  # docker
]
DUMP_DIR = None
for dir in DUMP_DIRS:
    if path_exists(dir):
        DUMP_DIR = dir
        print("Located data folder: %s" % DUMP_DIR)
        break
if DUMP_DIR is None:
    raise Exception("Unable to locate data folder.")

# MODEL DIRECTORY FOR SERVER
MODEL_FOLDER = DUMP_DIR + "models_server/"

# ESTIMATOR DIRECTORY
ESTIMATORS_DIR = DUMP_DIR + "estimators/"

# DATA DIRECTORY
DATA_DIR = DUMP_DIR + "data/"

# WIKIPEDIA DIRECTORY
WIKI_DIRS = ["/home/hertel/wikipedia/",
             "/data/1/matthias-hertel/wikipedia/"]
WIKI_DIR = None
for dir in WIKI_DIRS:
    if path_exists(dir):
        WIKI_DIR = dir
        break
if WIKI_DIR is None:
    WIKI_DIR = "__UNKNOWN_WIKI_DIRECTORY__"
    # print("WARNING: Unable to locate wikipedia folder.")

# wikipedia article IDs
WIKI_TRAINING_ARTICLE_IDS = WIKI_DIR + "training_article_ids.pkl"
WIKI_DEVELOPMENT_ARTICLE_IDS = WIKI_DIR + "development_article_ids.pkl"
WIKI_TEST_ARTICLE_IDS = WIKI_DIR + "test_article_ids.pkl"

# paragraphs
WIKI_PARAGRAPHS_DIR = WIKI_DIR + "single-file/"
WIKI_TRAINING_PARAGRAPHS = WIKI_PARAGRAPHS_DIR + "training_shuffled.txt"

# acl training data
ACL_TRAINING_FILE = DUMP_DIR + "acl_training.txt"

# BENCHMARKS DIRECTORY
BENCHMARKS_DIR = DUMP_DIR + "benchmarks/"

# sentence files
WIKI_SENTENCES_DIR = WIKI_DIR + "sentences/"
WIKI_TRAINING_SENTENCES = WIKI_SENTENCES_DIR + "training.txt"
WIKI_TRAINING_SENTENCES_SHUFFLED = WIKI_SENTENCES_DIR + "training_shuffled.txt"
WIKI_TUNING_SENTENCES = WIKI_SENTENCES_DIR + "tuning.txt"
WIKI_DEVELOPMENT_SENTENCES = WIKI_SENTENCES_DIR + "development.txt"
WIKI_TEST_SENTENCES = WIKI_SENTENCES_DIR + "test.txt"

# punkt tokenizer
WIKI_PUNKT_TOKENIZER = WIKI_DIR + "punkt_tokenizer.pkl"
EXTENDED_PUNKT_ABBREVIATIONS = WIKI_DIR + "extended_punkt_abbreviations.pkl"

# RESULTS DIRECTORY
RESULTS_DIR = DUMP_DIR + "results/"
RESULTS_DICT = RESULTS_DIR + "results.pkl"

# PLOTS DIRECTORY
PLOT_DIR = DUMP_DIR + "plots/"

# DICTIONARY DIRECTORY
DICT_FOLDER = DUMP_DIR + "dictionaries/"
# character frequencies
CHARACTER_FREQUENCY_DICT = DICT_FOLDER + "character_frequencies.pkl"
# old dictionaries for reproducibility
WIKI_ENCODER_DICT = DICT_FOLDER + "char2ix.pkl"
WIKI_DECODER_DICT = DICT_FOLDER + "ix2char.pkl"
# decision thresholds
DECISION_THRESHOLD_FILE = DICT_FOLDER + "new_decision_thresholds.pkl"
SINGLE_RUN_DECISION_THRESHOLD_FILE = DICT_FOLDER + "single_run_decision_thresholds.pkl"
TWO_PASS_DECISION_THRESHOLD_FILE = DICT_FOLDER + "two_pass_decision_thresholds.pkl"
LABELING_DECISION_THRESHOLD_FILE = DICT_FOLDER + "labeling_decision_thresholds.pkl"
# beam search penalties
BEAM_SEARCH_PENALTY_FILE = DICT_FOLDER + "beam_search_penalties.pkl"
TWO_PASS_BEAM_SEARCH_PENALTY_FILE = DICT_FOLDER + "two_pass_beam_search_penalties.pkl"
SEQ_ACC_BEAM_SEARCH_PENALTY_FILE = DICT_FOLDER + "beam_search_penalties_sequence_accuracy.pkl"
# token frequencies
TOKEN_FREQUENCY_DICT = DICT_FOLDER + "token_frequencies.pkl"
# unigrams
UNIGRAM_DELIM_FREQUENCY_DICT = DICT_FOLDER + "unigram_delim_frequencies.pkl"
UNIGRAM_NO_DELIM_FREQUENCY_DICT = DICT_FOLDER + "unigram_no_delim_frequencies.pkl"
MOST_FREQUENT_UNIGRAMS_DICT = DICT_FOLDER + "unigrams_most_frequent_%i.pkl"
# bigrams
BIGRAM_HOLDER = DICT_FOLDER + "bigram_holder.pkl"
# stump dict
STUMP_DICT = DICT_FOLDER + "token_stumps.pkl"
# tuning cases
CASES_FILE_CLEAN = DICT_FOLDER + "decision_cases_clean_%s%s.pkl"
CASES_FILE_NOISY = DICT_FOLDER + "decision_cases_noisy_%s%s.pkl"
# mixed encoder
MIXED_ENCODER_DICT = DICT_FOLDER + "mixed_encoder_dict.pkl"
# ocr errors
OCR_ERROR_FREQUENCIES_FILE = DICT_FOLDER + "ocr_error_frequencies.tsv"

# INTERMEDIATE RESULTS DIRECTORY
INTERMEDIATE_DIR = DUMP_DIR + "intermediate/"
THRESHOLD_FITTER_DIR = INTERMEDIATE_DIR + "threshold_fitter/"

# OPENAI
OPENAI_MODELS_FOLDER = None

"""for dir in [RESULTS_DIR, PLOT_DIR, INTERMEDIATE_DIR, THRESHOLD_FITTER_DIR, BENCHMARKS_DIR]:
    if not path_exists(dir):
        make_directory(dir)
        print("Made directory: %s" % dir)"""

# ACL CORPUS
ACL_CORPUS_DIR = DUMP_DIR + "acl_corpus/"
ACL_CORPUS_TRAINING_FILE = ACL_CORPUS_DIR + "training.txt"
ACL_ENCODER_DICT = ACL_CORPUS_DIR + "encoder_dict.pkl"

# ARXIV DATASET
ARXIV_BASE_DIR = "/home/hertel/tokenization-repair-dumps/claudius/"
ARXIV_GROUND_TRUTH_DIR = ARXIV_BASE_DIR + "groundtruth/"
PDF_EXTRACT_DIR = ARXIV_BASE_DIR + "PDFExtract/"
ARXIV_TRAINING_FILES = ARXIV_BASE_DIR + "training_files.txt"
ARXIV_DEVELOPMENT_FILES = ARXIV_BASE_DIR + "development_files.txt"
ARXIV_TEST_FILES = ARXIV_BASE_DIR + "test_files.txt"
ARXIV_TRAINING_LINES = ARXIV_BASE_DIR + "training_lines.txt"

ARXIV_CORPUS_DIR = DUMP_DIR + "arxiv_corpus/"
ARXIV_TRAINING_SEQUENCES = ARXIV_CORPUS_DIR + "training.txt"
ARXIV_ENCODER_DICT = ARXIV_CORPUS_DIR + "encoder_dict.pkl"


def benchmark_sub_directory(name: str,
                            subset: Subset,
                            subfolder: Optional[str]) -> str:
    subdir = name + "/" + subset.folder_name() + "/"
    if subfolder is not None:
        subdir += subfolder + "/"
    return subdir
