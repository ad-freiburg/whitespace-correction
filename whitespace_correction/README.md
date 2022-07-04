## Whitespace correction using Transformers

### Training instructions

In general, you will only need the following two commands:

First preprocess some data given a preprocessing configuration file using

`python -m whitespace_correction.preprocess_data --config <path_to_config_file>`

After that, train your model given a training configuration file using

`python -m whitespace_correction.train --config <path_to_config_file>`

To check the options and the required format of the preprocessing and training configuration files,
see [configs/data_preprocessing/base.yaml](../configs/data_preprocessing/base.yaml) and
[configs/train/base.yaml](../configs/train/base.yaml).

### Reproduce results

To reproduce the results of this project read the following sections `Download data`, `Preprocess data`
and `Model training`.

#### Download data

For our training data we used sequences from Wikipedia and Arxiv. We then created two versions of our training dataset,
one with spelling and OCR errors and one without.

To download the dataset without spelling and OCR errors execute the following commands:

```bash
# download raw text data as tar file and save it under data/raw
wget -P data/raw https://whitespace-correction.cs.uni-freiburg.de/training_mixed.txt.tar.gz
# extract tar file into directory data/raw/tokenization_repair_mixed
mkdir -p data/raw/whitespace_correction_mixed
tar -xzf data/raw/training_mixed.txt.tar.gz -C data/raw/whitespace_correction_mixed
# divide the data which is in one big .txt file into several smaller .txt files
# for faster data processing later on
python utils/subdivide_files.py \
  --files data/raw/whitespace_correction_mixed/training_mixed.txt \
  --lines-per-file 10000 \
  --out-dir data/raw/whitespace_correction_mixed_split
# convert all the .txt files into jsonl format, because that is required by the
# whitespace_correction library data preprocessing (might take a while)
python utils/txt_to_jsonl.py \
  --in-dir data/raw/whitespace_correction_mixed_split \
  --out-dir data/cleaned/whitespace_correction/mixed
```

To download the dataset with spelling and OCR errors execute the following commands:

```bash
# download raw text data as tar file and save it under data/raw
wget -P data/raw https://whitespace-correction.cs.uni-freiburg.de/training_mixed_ocr+spelling.txt.tar.gz
# extract tar file into directory data/raw/whitespace_correction_mixed_ocr_spelling_errors
mkdir -p data/raw/whitespace_correction_mixed_ocr_spelling_errors
tar -xzf data/raw/training_mixed_ocr+spelling.txt -C data/raw/whitespace_correction_mixed_ocr_spelling_errors
# divide the data which is in one big .txt file into several smaller .txt files
# for faster data processing later on
python utils/subdivide_files.py \
  --files data/raw/whitespace_correction_mixed_ocr_spelling_errors/training_mixed_ocr+spelling.txt \
  --lines-per-file 10000 \
  --out-dir data/raw/whitespace_correction_mixed_ocr_spelling_errors_split
# convert all the .txt files into jsonl format, because that is required by the
# whitespace_correction library data preprocessing (might take a while)
python utils/txt_to_jsonl.py \
  --in-dir data/raw/whitespace_correction_mixed_ocr_spelling_errors_split \
  --out-dir data/cleaned/whitespace_correction/mixed_with_errors
```

You are now ready to use the generated .jsonl files to preprocess the LMDB datasets for model training.

#### Preprocess data

All data preprocessing config files can be found in
[configs/data_preprocessing/whitespace_correction](configs/data_preprocessing/whitespace_correction).

To preprocess the data use the following command and replace
`<path_to_config_file>` with the preprocessing config file you want:

`python -m whitespace_correction.preprocess_data --config <path_to_config_file>`

If you e.g. want to preprocess the Arxiv dataset with spelling and OCR errors for the EO models then execute:

`python -m whitespace_correction.preprocess_data --config configs/data_preprocessing/whitespace_correction/whitespace_correction_char_eo_arxiv_with_errors.yaml`

Or if you e.g. want to preprocess the Arxiv dataset without spelling and OCR errors dataset for the NMT models then
execute:

`python -m whitespace_correction.preprocess_data --config configs/data_preprocessing/whitespace_correction/whitespace_correction_char_nmt_arxiv_no_errors.yaml`

You will need to set environment variables for some configs to work, but you will get error messages when the variables
are not set.

#### Model training

All training config files can be found in
`configs/train/whitespace_correction`.

To train a model use the following command and replace
`<path_to_config_file>` with the preprocessing config file you want:

`python -m whitespace_correction.train --config <path_to_config_file>`

If you e.g. want to train an EO model then execute:

`python -m whitespace_correction.train --config configs/train/whitespace_correction/eo.yaml`

Or if you e.g. want to train a NMT model then execute:

`python -m whitespace_correction.train --config configs/train/whitespace_correction/nmt.yaml`

> The training code only works for single or multi GPU training on a single node, so if you do want to train on
> more than one node, for now you will need to rewrite the code to support it.

> Be aware that the models from the paper were trained on 8 Nvidia RTX 2080 Ti GPUs for 3 epochs,
> so if you do not have similar compute resources available you might not be able to exactly reproduce the
> results from the paper.

You will need to set environment variables for some configs to work, but you will get error messages when the variables
are not set.

Instead of setting the environment variables, you can of course also copy the training config file and overwrite the
values directly.

| Env variable      | eo_large_arxiv_with_errors | eo_medium_arxiv_with_errors | eo_small_arxiv_with_errors | 
|-------------------|----------------------------|-----------------------------|----------------------------|
| MODEL_NAME        | eo_large_arxiv_with_errors | eo_medium_arxiv_with_errors | eo_small_arxiv_with_errors |
| LMDB_PATH         | <path_to_lmdb>             | <path_to_lmdb>              | <path_to_lmdb>             |
| BATCH_MAX_TOKENS* | 8192                       | 8192                        | 8192                       |
| NUM_EPOCHS        | 3                          | 3                           | 3                          |
| NUM_LAYERS        | 12                         | 6                           | 3                          |

*The number of tokens per batch is set per training process / GPU, so if you use a different number of GPUs than 8,
adjust this setting accordingly to keep the overall tokens per batch about the same.
