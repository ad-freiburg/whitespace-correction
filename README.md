## Whitespace correction using Transformers

Correct missing or spurious whitespaces in text.

### Installation

This project is mainly tested with Python 3.10, but should work fine with Python 3.8 and newer versions.

#### From PyPI

Windows (x64) and  Linux are currently supported when installing from PyPI.

```bash
pip install whitespace-correction
```

#### From source

```bash
git clone git@github.com:ad-freiburg/whitespace-correction.git
cd whitespace-correction
pip install .

```

### Usage

#### From python

```python
from whitespace_correction import WhitespaceCorrector

# list all available models
print(WhitespaceCorrector.available_models())

# create a whitespace corrector instance, using the default pretrained model
cor = WhitespaceCorrector.from_pretrained()
# you can move the whitespace correction model to a different device, e.g.
cor.to("cuda")  # default
cor.to("cuda:3")  # if you have multiple GPUs (alternatively use cor.to(3))
cor.to("cpu")
# you can also set the inference precision (default is the 
# precision used for training), e.g.
cor.set_precision("fp32")
cor.set_precision("fp16")
cor.set_precision("bfp16")

# correct single strings
corrected_string = cor.correct_text("p l e se,repiar thissen ten se!")

# correct multiple strings at once
repaired_strings = cor.correct_text([
    "p l e se,repiar thissen ten se!",
    "alsosplitthissentenceforme"
])

# correct text file (every line is treated as a separate sequence to correct),
# returns an iterator over corrected lines
corrected_lines = cor.correct_file("path/to/file.txt")
# optionally specify an output file,
# returns None
corrected_lines = cor.correct_file("path/to/file.txt", output_file="save/output/here.txt")
```

#### From command line

After installation the command `wsc` (short for **w**hite**s**pace **c**orrection) is available in your python environment. 
It lets you use the whitespace correction models directly from the command line.
Below are examples of how to use `wsc`. See `wsc -h` for all options.

```bash
# print version
wsc -v

# list available models
wsc -l

# by default wsc tries to read stdin, corrects the input it got line by line 
# and prints the corrected lines back out
# therefore, you can for example use whitespace correction with pipes
echo "splitthissentenceforme" | wsc
cat "path/to/input/file.txt" | wsc > output.txt

# correct a string using
wsc -p "splitthissentenceforme"

# correct a text file line by line and print the corrected lines
wsc -f path/to/input/file.txt
# optionally specify an output file path where the corrected lines are saved
wsc -f path/to/input/file.txt -o output.txt

# start an interactive whitespace correction session
# where your input will be corrected and printed back out
wsc -i

# start a whitespace correction server with the following endpoints:
### /models [GET] --> output: available models as json 
### /info [GET] --> output: info about backend as json
### /evaluate [POST] input: input, output, and groundtruth text --> output: evaluation metrics as json
### /correct [POST] input: some text to correct --> output: corrected text and runtime information as json
wsc --server <config_file>

### OPTIONS
### Pass the following flags to the wsc command to customize its behaviour
-m <model_name> # use a different whitespace correction model than the default one 
--cpu # force execution on CPU, by default a GPU is used if available
--progress # display a progress bar (always on when a file is repaired using -f)
-b <batch_size> # specify a different batch size
-batch-max-tokens <batch_max_tokens> # limit batch by a number of tokens and not by number of samples
-u # do not sort the inputs before correcting
--precision # set inference precision (one of fp32, fp16 and bfp16)
-e <experiment_dir> # specify the path to an experiment directory to load the model from 
                    # (equivalent to WhitespaceCorrector.from_experiment(experiment_dir) in python API)
--force-download # force download of the whitespace correction model even if it was already downloaded
--progress # show a progress bar while correcting
--report # print a report on the runtime of the model after finishing the correction
```

> Note: When first using `wsc` with a pretrained model, the model needs to be downloaded, so depending on
> your internet speed the command might take considerably longer.

> Note: Loading the whitespace correction model requires an initial startup time each time you
> invoke the `wsc` command. CPU startup time is around 1s, GPU startup time around 3.5s, so for small
> inputs or files you should probably pass the `--cpu` flag to force CPU execution for best performance.

> See [configs/server.yaml](configs/server.yaml) for an exemplary server configuration file.

### Documentation

#### Use pretrained model

If you just want to use this project to correct whitespaces, this is the recommended way.

```python
from whitespace_correction import WhitespaceCorrector

cor = WhitespaceCorrector.from_pretrained(
    # pretrained model to load, get all available models from available_models(),
    # if None, loads the default model
    model=None,
    # the device to run the model on
    # ("cuda" by default)
    device="cuda",
    # optional path to a cache directory where downloaded models will be extracted to,
    # if None, we check the env variable WHITESPACE_CORRECTION_CACHE_DIR, if it is not set 
    # we use a default cache directory at <install_path>/api/.cache 
    # (None by default)
    cache_dir=None,
    # optional path to a download directory where pretrained models will be downloaded to,
    # if None, we check the env variable WHITESPACE_CORRECTION_DOWNLOAD_DIR, if it is not set 
    # we use a default download directory at <install_path>/api/.download
    # (None by default)
    download_dir=None,
    # force download of model even if it already exists in download dir
    # (False by default)
    force_download=False
)
```

When used for the first time with the command line interface or Python API the pretrained model will be automatically downloaded. 
However, you can also download our pretrained models first as zip files, put them in a directory on your local drive 
and set `WHITESPACE_CORRECTION_DOWNLOAD_DIR` (or the `download_dir` parameter above) to this directory.

Download links:
- [eo_large_char_v1](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_large_char_v1.zip)
- [eo_large_char](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_large_char_v2.zip)
- [eo_large_byte](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_large_byte_v2.zip)
- [eo_medium_char_v1](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_medium_char_v1.zip)
- [eo_medium_char](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_medium_char_v2.zip)
- [eo_medium_byte](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_medium_byte_v2.zip)
- [eo_larger_byte](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/eo_huge_byte_v2.zip)
- [ed_large_char](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/ed_large_v1.zip)
- [ed_medium_char](https://ad-publications.informatik.uni-freiburg.de/ACL_whitespace_correction_transformer_BHW_2023.materials/ed_medium_v1.zip)

#### Use own model

Once you trained your own model you can use it in the following way.

```python
from whitespace_correction import WhitespaceCorrector

cor = WhitespaceCorrector.from_experiment(
    # path to the experiment directory that is created by your training run
    experiment_dir="path/to/experiment_dir",
    # the device to run the model on
    # ("cuda" by default)
    device="cuda"
)
```

### Directory structure

The most important directories you might want to look at are:

```
configs -> (example yaml config files for training and server)
src -> (library code used by this project)
```

### Docker

You can also run this project using docker. Build the image using

`docker build -t whitespace-correction .`

If you have an older GPU build the image using

`docker build -t whitespace-correction -f Dockerfile.old .`

By default, the entrypoint is set to the `wsc` command, 
so you can use the Docker setup like described [here](#from-command-line) earlier.

You can mount /wsc/cache and /wsc/download to volumes on your machine, such that
you do not need to download the models every time.

```bash
# correct text
docker run whitespace-correction -c "correctthisplease"

# correct file
docker run whitespace-correction -f path/to/file.txt

# start a server
docker run whitespace-correction --server path/to/config.yaml

# with volumes
docker run -v $(pwd)/.cache:/wsc/cache -v $(pwd)/.download:/wsc/download \
  whitespace-correction -c "correctthisplease"

# optional parameters recommended when using a GPU:
# --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

Note
----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support.
```
