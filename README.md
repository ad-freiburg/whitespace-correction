## Tokenization repair using Transformers

Repair missing or spurious whitespaces in text.

### Installation

This project is mainly tested with Python 3.8, but should work fine with Python 3.6 and newer versions

#### From PyPi

```bash
pip install whitespace-repair
```

#### From source

```bash
git clone git@github.com:ad-freiburg/trt.git
cd trt

# if you just want to use pretrained models
pip install .
# alternatively, if you also want to train your own models
pip install .[train]

```

### Usage

#### From python

```python
from whitespace_repair import TokenizationRepairer, get_available_models

# list all available models
print(get_available_models())

# create a tokenization repair instance, using the default pretrained model
tok_rep = TokenizationRepairer.from_pretrained()
# you can move the tokenization repair model to a different device, e.g.
tok_rep.to("cuda")  # default
tok_rep.to("cuda:3")  # if you have multiple GPUs (alternatively use tok_rep.to(3))
tok_rep.to("cpu")
# you can also set the inference precision, e.g.
tok_rep.set_precision("fp32")  # default
tok_rep.set_precision("fp16")
tok_rep.set_precision("bfp16")

# repair single strings
repaired_string = tok_rep.repair_text("p l e se,repiar thissen ten se!")

# repair multiple strings at once
repaired_strings = tok_rep.repair_text([
    "p l e se,repiar thissen ten se!",
    "alsosplitthissentenceforme"
])

# repair text file (every line is treated as a separate sequence to repair)
repaired_lines = tok_rep.repair_file("path/to/file.txt")
# optionally specify a output file
repaired_lines = tok_rep.repair_file("path/to/file.txt", output_file_path="save/output/here.txt")
```

#### From command line

After installation the command `wr` is available in your python environment. It lets you use the tokenization repair
models directly from the command line. Below are examples of how to use `wr`. See `wr -h` for all options.

```bash
# print version
wr -v

# list available models
wr -l

# by default whitespace_repair tries to read stdin, repairs the input it got line by line 
# and prints the repaired lines back out
# therefore, you can for example use whitespace_repair with pipes
echo "splitthissentenceforme" | wr
cat "path/to/input/file.txt" | wr > output.txt

# repair a string using
wr -r "splitthissentenceforme"

# repair a text file line by line and print the repaired lines
wr -f path/to/input/file.txt
# optionally specify an output file path where the repaired lines are saved
wr -f path/to/input/file.txt -o output.txt

# start an interactive tokenization repair session
# where your input will be repaired and printed back out
wr -i

# start a tokenization repair server with the following endpoints:
### /models [GET] --> list available models as json 
### /repair_text?text=texttorepair [GET] --> repaired text as json
### /repair_file [POST] --> repaired file as json
### To specify which model to use, you can use the model query parameter 
### (e.g. /repair_file?model=eo_small_arxiv_with_errors), default model is eo_large_arxiv_with_errors
wr --server <config_file>

### OPTIONS
### Pass the following flags to the whitespace_repair command to customize its behaviour
-m <model_name> # use a different tokenization repair model than the default one 
--cpu # force execution on CPU, by default a GPU is used if available
--progress # display a progress bar (always on when a file is repaired using -f)
-b <batch_size> # specify a different batch size
-u # do not sort the inputs before repairing
-p # switch on pipe mode (useful but not needed when using Linux pipes)
--precision # set inference precision (one of fp32, fp16 and bfp16)
-e <experiment_dir> # specify the path to an experiment directory to load the model from 
                    # (equivalent to TokenizationRepairer.from_experiment(experiment_dir) in python API)
--force-download # force download of the tokenization repair model even if it was already downloaded
--report <file_path> # save a report on the runtime of the model in form of a markdown table in a file
```

> Note: When first using `wr` with a pretrained model, the model needs to be downloaded, so depending on
> your internet speed the command might take considerably longer.

> Note: Loading the tokenization repair model requires an initial startup time each time you
> invoke the `wr` command. CPU startup time is around 1s, GPU startup time around 3.5s, so for small
> inputs or files you should probably pass the `-c` flag to force CPU execution for best performance.

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

If you just want to use this project to repair tokenization, this is the recommended way. See the available model for
the names of all available

```python
from whitespace_repair import TokenizationRepairer, get_available_models

tok_rep = TokenizationRepairer.from_pretrained(
    # pretrained model to load, get all available models from get_available_models() 
    # (eo_arxiv_with_errors by default)
    model="eo_arxiv_with_errors",
    # whether to use a GPU if available or not 
    # (True by default)
    use_gpu=True,
    # optional path to a cache directory where pretrained models will be downloaded to,
    # if None, we check the env variable TOKENIZATION_REPAIR_CACHE_DIR, if it is not set 
    # we use a default cache directory at <wr_install_path>/api/.cache 
    # (None by default)
    cache_dir=None,
)
```

#### Use own model

Once you trained you model you can use it in the following way

```python
from whitespace_repair import TokenizationRepairer

tok_rep = TokenizationRepairer.from_experiment(
    # path to the experiment directory that is created by your training run
    experiment_dir="path/to/experiment_dir",
    # whether to use a GPU if available or not 
    # (True by default)
    use_gpu=True
)
```

See [tokenization_repair/README.md](tokenization_repair/README.md) for instructions to train you own model.

### Directory structure

The most important directories you might want to look at are:

```
configs -> (example yaml config files for data preprocessing, models and training)
src -> (library code used by this project)
tests -> (unit tests for whitespace_repair library)
tokenization_repair -> (actual tokenization repair project directory)
    benchmark   -> (benchmarks, benchmark results and functionality to run and evaluate benchmarks)
    configs     -> (yaml config files used to preprocess data and 
                    train the tokenization repair models)
```

### Docker

You can also run this project using docker. Build the image using

`docker build -t tokenization_repair_transformers .`

After that you can run the project using

```
docker run -it [--gpus all] tokenization_repair_transformers

Note
-----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support. You also need to add '--gpus all' 
to the 'docker run' command from above.
```

Inside the container the [`wr` command](#from-command-line) is available to you.

If you e.g. have some text files you want to repair using the pretrained models start your container using
`docker run --gpus all -v <directory_path>:/text_files -it tokenization_repair_transformers` where `<directory_path>`
contains the text files. Then inside the container you can repair them
with `wr -f /text_files/file_1.txt -o /text_files/file_1_repaired.txt`.

You can also start a tokenization repair server inside the container using `wr --server <config_file>`. Keep in mind
that if you want to access the server from outside the container you have to expose the port using
`docker run --gpus all -p <outside_port>:<server_port> -it tokenization_repair_transformers`.
