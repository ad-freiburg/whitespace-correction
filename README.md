## Tokenization repair using Transformers

### Installation

This project is mainly tested with Python 3.8, but should work fine with Python 3.7 and newer versions

```bash
git clone git@github.com:bastiscode/trt.git
cd trt

# if you just want to use pretrained models
pip install .
# alternatively, if you also want to train your own models
pip install .[train]

# additionally, if you also want to run the demo
pip install -r tokenization_repair/requirements.txt
```

### Usage

```python
from trt import TokenizationRepairer
tok_rep = TokenizationRepairer.from_pretrained()

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

List available models
```python
from trt import get_available_models
print(get_available_models())
```

### Documentation

#### Use pretrained model
If you just want to use this project to repair tokenization, this is the
recommended way. See the available model for the names of all available
```python
from trt import TokenizationRepairer, get_available_models
tok_rep = TokenizationRepairer.from_pretrained(
    # pretrained model to load, get all available models from get_available_models() 
    # (eo_arxiv_with_errors by default)
    model="eo_arxiv_with_errors", 
    # whether to use a GPU if available or not 
    # (True by default)
    use_gpu=True, 
    # optional path to a cache directory where pretrained models will be downloaded to,
    # if None, we check the env variable TOKENIZATION_REPAIR_CACHE_DIR, if it is not set 
    # we use a default cache directory at <trt_install_path>/api/.cache 
    # (None by default)
    cache_dir=None, 
)
```

#### Use own model
Once you trained you model you can use it in the following way
```python
from trt import TokenizationRepairer
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
trt -> (library code used by this project)
tests -> (unit tests for trt library)
tokenization_repair -> (actual tokenization repair project directory)
    demo        -> (tokenization repair streamlit demo)
    benchmark   -> (benchmarks, benchmark results and functionality to run and evaluate benchmarks)
    configs     -> (yaml config files used to preprocess data and 
                    train the tokenization repair models)
```

### Docker

You can also run this project using docker. Build the image using 

`docker build -t tokenization_repair_transformers .`

After that you can run the project using

```
docker run -it [--gpus all] 
-p <local_port>:8501 
-v <experiments>:/trt/tokenization_repair/experiments 
tokenization_repair_transformers

where 
-----
<local_port> is the port you want to access the demo on in your browser
<experiments> is the path to a directory where your training experiments are stored

Note
-----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support. You also need to add '--gpus all' 
to the 'docker run' command from above.
```

### Demo

To view the demo, start the project in a docker container using the instructions from above
and execute `make demo`.
