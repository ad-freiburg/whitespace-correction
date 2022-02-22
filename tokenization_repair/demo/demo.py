import argparse
import glob
import os
from typing import List

import streamlit as st

from pages.evaluate_benchmarks import show_evaluate_benchmarks
from pages.home import show_home
from pages.info import show_info
from pages.interactive import show_interactive
from pages.model_spec import show_model_spec
from pages.run_benchmarks import show_run_benchmarks
from pages.upload_benchmarks import show_upload_benchmarks
from pages.visualization import show_visualization
from utils import load_model

import torch

from trt.model.transformer import get_tokenizers_from_model
from trt.utils import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments",
                        required=True,
                        help="Path to the experiments dir")
    parser.add_argument("-b", "--benchmarks",
                        required=True,
                        help="Path to the benchmarks dir")
    parser.add_argument("-r", "--results",
                        required=True,
                        help="Path to the results dir")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    BASE_DIR = os.path.dirname(__file__)
    args = parse_args()

    st.markdown(f"""
                <style>
                    .reportview-container .main .block-container{{
                        max-width: {1200}px;
                    }}
                </style>
                """,
                unsafe_allow_html=True)

    st.write("""
    # Tokenization repair using Transformers
    ###### *by Sebastian Walter, last updated on 2021/03/28*
    ---
    """)

    st.sidebar.write("# Navigation")
    st.sidebar.write("##### Select a page")
    select_todo = st.sidebar.selectbox("Pages",
                                       ["Home",
                                        "Interactive Demo",
                                        "Model specification",
                                        "Upload benchmarks",
                                        "Run benchmarks",
                                        "Evaluate on benchmarks",
                                        "Visualize",
                                        "Info"])

    st.sidebar.write("---")
    st.sidebar.write("# Options")

    experiments = glob.glob(f"{args.experiments}/*")
    configs: List[config.Config] = [config.Config.from_yaml(os.path.join(exp, "config.yaml"))
                                    for exp in experiments]

    # assert len(set(cfg.model.name for cfg in configs)) == len(configs), "all configs must specify a model with a " \
    #                                                                     "unique model name"

    for cfg in configs:
        cfg.model.pretrained = None

        if cfg.model.type == "encoder_with_head" or cfg.model.type == "transformer":
            cfg.model.encoder.pretrained = None

        if cfg.model.type == "decoder" or cfg.model.type == "transformer":
            cfg.model.decoder.pretrained = None

    select_models = {}
    for i, c in enumerate(configs):
        checkpoint_path = os.path.abspath(
            os.path.join(experiments[i],
                         "checkpoints",
                         f"{c.model.name}-checkpoint-best.pt"))
        experiment_dir = experiments[i].split("/")[-1]
        info = (checkpoint_path, experiment_dir, c)

        if c.model.name in select_models:
            select_models[f"{c.model.name}_{os.path.dirname(experiment_dir)}"] = info
        else:
            select_models[c.model.name] = info

    st.sidebar.write("##### Select the model you want to use")
    checkpoint = st.sidebar.selectbox("Transformer models",
                                      sorted(list(select_models.keys())),
                                      index=0)

    checkpoint_path, exp_dir, config = select_models[checkpoint]  # type: str, str, config.Config

    if checkpoint:
        st.sidebar.write(f"*`Experiment directory: {exp_dir}`*")

    st.sidebar.write("##### Select a device to run the model on")
    num_gpus = torch.cuda.device_count()
    gpu_names = [f"{torch.cuda.get_device_name(i)} (cuda:{i})" for i in range(num_gpus)]
    select_device = st.sidebar.selectbox("Devices",
                                         [-1] + list(range(num_gpus)),
                                         format_func=lambda i: "CPU" if i == -1 else gpu_names[i])
    device = torch.device("cpu" if select_device < 0 else f"cuda:{select_device}")

    st.sidebar.write("##### Select an inference method")
    select_inference_method = st.sidebar.selectbox("Inference methods",
                                                   ["greedy", "sample", "beam"])
    sample_topk = None
    beam_width = None
    inference_kwargs = {
        "inference_method": select_inference_method,
        "sample_top_k": sample_topk,
        "beam_width": beam_width
    }

    if select_inference_method == "sample":
        sample_topk = st.sidebar.slider("Sample from top k predictions",
                                        min_value=1,
                                        max_value=9,
                                        value=5)
        inference_kwargs["sample_top_k"] = sample_topk
    elif select_inference_method == "beam":
        beam_width = st.sidebar.slider("Specify the beam width",
                                       min_value=2,
                                       max_value=9,
                                       value=3)
        inference_kwargs["beam_width"] = beam_width

    model, steps = load_model(config.model, checkpoint_path, device)
    encoder_tokenizer, decoder_tokenizer = get_tokenizers_from_model(model)

    if select_todo == "Home":
        show_home()

    elif select_todo == "Info":
        show_info(base_dir=os.path.join(BASE_DIR, "..", ".."))

    elif select_todo == "Model specification":
        show_model_spec(config=config, steps=steps, model=model)

    elif select_todo == "Interactive Demo":
        show_interactive(model=model, model_type=config.model.type, **inference_kwargs)

    elif select_todo == "Upload benchmarks":
        show_upload_benchmarks(benchmarks_dir=args.benchmarks)

    elif select_todo == "Run benchmarks":
        show_run_benchmarks(model=model,
                            model_config=config.model,
                            benchmarks_dir=args.benchmarks,
                            results_dir=args.results,
                            **inference_kwargs)

    elif select_todo == "Evaluate on benchmarks":
        show_evaluate_benchmarks(benchmarks_dir=args.benchmarks,
                                 results_dir=args.results)

    elif select_todo == "Visualize":
        show_visualization(model=model, model_type=config.model.type, **inference_kwargs)
