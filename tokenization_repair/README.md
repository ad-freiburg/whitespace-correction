## Masters project - Tokenization repair using Transformers
*by Sebastian Walter*

> *All the commands on this page should be executed 
inside the Docker container (for infos about the Docker setup see `README.md` in parent directory)*

### Training instructions

In general, you will only need the following two commands:

First preprocess some data given a preprocessing configuration file using

`python -m tfnm_text.preprocess_data --config <path_to_config_file>`

After that, train your model given a training configuration file using

`python -m tfnm_text.train --config <path_to_config_file>`

To check the options and the required format of the preprocessing and training configuration files, 
see `/masters_project/configs/data_preprocessing/base.yaml` and `/masters_project/configs/train/base.yaml`.

### Reproduce results

To reproduce the results of this project (see blog post) 
read the following sections `Download data`, `Preprocess data` and `Model training`.

> The two sections `Download data` and `Preprocess data` are just here for reference. It
> is recommended that you skip these two sections and just mount the data from 
> `/nfs/students/sebastian-walter/data` into the Docker container 
> under `/masters_project/tokenization_repair/data` (cf. README.md in parent directory 
> or command at the end of the Dockerfile).
> This not only saves a lot of time, but also ensures you are training with the exact same data.

#### Download data

> *Skip this if you have already setup the data and/or mounted it into the 
Docker container*

Download the Wikipedia and Bookcorpus datasets using

`make download_data`

Clean and format the datasets using

`python setup_data.py -i data -o data/cleaned -d wiki` and 
`python setup_data.py -i data -o data/cleaned -d bookcorpus`

#### Preprocess data

> *Skip this if you have already setup the data and/or mounted it into the 
Docker container*

All data preprocessing config files can be found in 
`configs/data_preprocessing/tokenization_repair`.

To preprocess the data use the following command and replace 
`<path_to_config_file>` with the preprocessing config file you want:

`python -m tfnm_text.preprocess_data --config <path_to_config_file>`

If you e.g. want to preprocess the Wikipedia dataset for the EO models then execute:

`python -m tfnm_text.preprocess_data --config configs/data_preprocessing/tokenization_repair/tokenization_repair_char_wiki.yaml`

Or if you e.g. want to preprocess the Wikipedia dataset for the NMT models then execute:

`python -m tfnm_text.preprocess_data --config configs/data_preprocessing/tokenization_repair/tokenization_repair_char_nmt_wiki.yaml`

> *You will need to set environment variables for some configs to work. 
> You will get error messages when the variables are not set.*

#### Model training

All training config files can be found in 
`configs/train/tokenization_repair`.

To train a model use the following command and replace 
`<path_to_config_file>` with the preprocessing config file you want:

`python -m tfnm_text.train --config <path_to_config_file>`

If you e.g. want to train an EO model then execute:

`python -m tfnm_text.train --config configs/train/tokenization_repair/eo.yaml`

Or if you e.g. want to a NMT model then execute:

`python -m tfnm_text.train --config configs/train/tokenization_repair/nmt.yaml`

> *You will need to set environment variables for some configs to work. 
> You will get error messages when the variables are not set. \
> Look in the blog post for some guidance on what values to set for the env variables. If you e.g. want to train 
> an EO-small model then set the env variable NUM_LAYERS=2. \
> You can also inspect the `config.yaml` files 
> for every experiment in the `experiments` directory that you mounted from `/nfs/students/sebastian-walter/experiments`
> when starting the Docker container. These `config.yaml` files are the actual files used
> to achieve the results mentioned in the blog post. Be aware that the models from the blog post
> were trained on 2 - 8 Nvidia Tesla V100 GPUs for 4 epochs max, so if you do not have similar compute resources 
> available you might not be able to exactly match the results from the blog post.*
