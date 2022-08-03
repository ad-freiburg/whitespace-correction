## Whitespace correction using Transformers

Correct missing or spurious whitespaces in text.

### Installation

This project is mainly tested with Python 3.8, but should work fine with Python 3.7 and newer versions

#### From PyPi

```bash
pip install whitespace-correction
```

#### From source

```bash
git clone git@github.com:ad-freiburg/whitespace-correction.git
cd whitespace-correction

# if you just want to use pretrained models
pip install .
# alternatively, if you also want to train your own models
pip install .[train]

```

### Usage

#### From python

```python
from whitespace_correction import WhitespaceCorrector, get_available_models

# list all available models
print(get_available_models())

# create a whitespace corrector instance, using the default pretrained model
ws_cor = WhitespaceCorrector.from_pretrained()
# you can move the whitespace correction model to a different device, e.g.
ws_cor.to("cuda")  # default
ws_cor.to("cuda:3")  # if you have multiple GPUs (alternatively use ws_cor.to(3))
ws_cor.to("cpu")
# you can also set the inference precision, e.g.
ws_cor.set_precision("fp32")  # default
ws_cor.set_precision("fp16")
ws_cor.set_precision("bfp16")

# correct single strings
corrected_string = ws_cor.correct_text("p l e se,repiar thissen ten se!")

# correct multiple strings at once
repaired_strings = ws_cor.correct_text([
    "p l e se,repiar thissen ten se!",
    "alsosplitthissentenceforme"
])

# correct text file (every line is treated as a separate sequence to correct)
corrected_lines = ws_cor.correct_file("path/to/file.txt")
# optionally specify an output file
corrected_lines = ws_cor.correct_file("path/to/file.txt", output_file_path="save/output/here.txt")
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
wsc -c "splitthissentenceforme"

# correct a text file line by line and print the corrected lines
wsc -f path/to/input/file.txt
# optionally specify an output file path where the corrected lines are saved
wsc -f path/to/input/file.txt -o output.txt

# start an interactive whitespace correction session
# where your input will be corrected and printed back out
wsc -i

# start a whitespace correction server with the following endpoints:
### /models [GET] --> list available models as json 
### /info [GET] --> info about backend as json
### /correct_text [POST] --> corrected text and runtime information as json
### To specify which model to use, you can use the model query parameter 
### (e.g. /correct_text?model=eo_small), default model is eo_large
wsc --server <config_file>

### OPTIONS
### Pass the following flags to the wsc command to customize its behaviour
-m <model_name> # use a different whitespace correction model than the default one 
--cpu # force execution on CPU, by default a GPU is used if available
--progress # display a progress bar (always on when a file is repaired using -f)
-b <batch_size> # specify a different batch size
-u # do not sort the inputs before correcting
-p # switch on pipe mode (useful but not needed when using Linux pipes)
--precision # set inference precision (one of fp32, fp16 and bfp16)
-e <experiment_dir> # specify the path to an experiment directory to load the model from 
                    # (equivalent to WhitespaceCorrector.from_experiment(experiment_dir) in python API)
--force-download # force download of the whitespace correction model even if it was already downloaded
--report <file_path> # save a report on the runtime of the model in form of a markdown table in a file
```

> Note: When first using `wsc` with a pretrained model, the model needs to be downloaded, so depending on
> your internet speed the command might take considerably longer.

> Note: Loading the whitespace correction model requires an initial startup time each time you
> invoke the `wsc` command. CPU startup time is around 1s, GPU startup time around 3.5s, so for small
> inputs or files you should probably pass the `--cpu` flag to force CPU execution for best performance.

> Note: The server configuration file must contain a json object with the following keys:
> - host (str, required)
> - port (int, required)
> - timeout (float, optional, by default timeout is set to 10 seconds)
> - models (list of str, optional, by default all models will be served)
> - precision (str, optional, by default precision is set to fp32)
> 
> See [configs/server.json](configs/server.json) for an exemplary server configuration file

### Documentation

#### Use pretrained model

If you just want to use this project to correct whitespaces, this is the recommended way.

```python
from whitespace_correction import WhitespaceCorrector, get_available_models

ws_cor = WhitespaceCorrector.from_pretrained(
    # pretrained model to load, get all available models from get_available_models() 
    # (eo_large by default)
    model="eo_large",
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
- [eo_large](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/eo_large.zip)
- [eo_medium](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/eo_medium.zip)
- [eo_small](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/eo_small.zip)
- [ed_large](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/ed_large.zip)
- [ed_medium](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/ed_medium.zip)
- [ed_small](https://ad-publications.informatik.uni-freiburg.de/EMNLP_whitespace_correction_transformer_BHW_2022.materials/ed_small.zip)

#### Use own model

Once you trained your own model you can use it in the following way.

```python
from whitespace_correction import WhitespaceCorrector

ws_cor = WhitespaceCorrector.from_experiment(
    # path to the experiment directory that is created by your training run
    experiment_dir="path/to/experiment_dir",
    # the device to run the model on
    # ("cuda" by default)
    device="cuda"
)
```

See [whitespace_correction/README.md](whitespace_correction/README.md) for instructions to train your own model.

### Directory structure

The most important directories you might want to look at are:

```
configs -> (example yaml config files for data preprocessing, models and training)
src -> (library code used by this project)
tests -> (unit tests for whitespace_correction library)
whitespace_correction -> (actual whitespace correction project directory)
    benchmark   -> (benchmarks, benchmark results and functionality to run and evaluate benchmarks)
    configs     -> (yaml config files used to preprocess data and 
                    train the whitespace correction models)
```

### Docker

You can also run this project using docker. Build the image using

`docker build -t whitespace-correction .`

After that you can run the project using

```
docker run --name wsc -it [--gpus all] --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 whitespace-correction

Note
-----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support. You also need to add '--gpus all' 
to the 'docker run' command from above.
```

Inside the container the [`wsc` command](#from-command-line) is available to you.

If you e.g. have some text files you want to repair using the pretrained models start your container using
`docker run --name wsc --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v <directory_path>:/text_files -it whitespace-correction` where `<directory_path>`
contains the text files. Then inside the container you can repair them
with `wsc -f /text_files/file_1.txt -o /text_files/file_1_corrected.txt`.

You can also start a tokenization repair server inside the container using `wsc --server <config_file>`. Keep in mind
that if you want to access the server from outside the container you have to expose the port using
`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p <outside_port>:<server_port> -it whitespace-correction`.
