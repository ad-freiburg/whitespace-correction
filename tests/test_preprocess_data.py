import os
import subprocess
import tempfile

import pytest

from trt.utils import config
from trt.utils.config import PreprocessingConfig

BASE_DIR = os.path.dirname(__file__)


@pytest.mark.parametrize("tokenizer_name", ["char", "byte", "tokenization_repair"])
@pytest.mark.parametrize("target_tokenizer_name", ["char", "byte", "tokenization_repair"])
@pytest.mark.parametrize("pretokenize", [True, False])
@pytest.mark.parametrize("preprocessing_config", ["edit_tokens_corruption", "tokenization_repair"], indirect=True)
class TestDataPreprocessing:
    def test_data_preprocessing(
            self,
            tokenizer_name: str,
            target_tokenizer_name: str,
            pretokenize: bool,
            preprocessing_config: PreprocessingConfig
    ) -> None:
        lmdb_name = f"dummy_lmdb_{tokenizer_name}_{target_tokenizer_name}_{pretokenize}_{preprocessing_config.type}"

        with tempfile.TemporaryDirectory() as temp_dir:
            data_preprocessing_config = config.DataPreprocessingConfig(
                data=[os.path.join(BASE_DIR, "data", "dummy.jsonl")],
                seed=22,
                output_dir=temp_dir,
                tokenizer=tokenizer_name,
                target_tokenizer=target_tokenizer_name,
                pretokenize=pretokenize,
                ensure_equal_length=True,
                preprocessing=[preprocessing_config],
                lmdb_name=lmdb_name,
                max_sequences=1000,
                max_sequence_length=512,
                cut_overflowing=False
            )

            # save the config in yaml format
            config_path = os.path.join(temp_dir, f"{lmdb_name}_config.yaml")
            with open(config_path, "w", encoding="utf8") as cfg_file:
                cfg_file.write(str(data_preprocessing_config))

            # call the preprocessing command as one does from the command line
            cmd = f"python -m trt.preprocess_data --config {config_path}"
            p = subprocess.Popen(cmd, shell=True)
            exit_code = p.wait()

            # check that exit code is 0 and the lmdb database exists in the temp directory
            assert exit_code == 0
            assert os.path.exists(os.path.join(temp_dir, lmdb_name))
