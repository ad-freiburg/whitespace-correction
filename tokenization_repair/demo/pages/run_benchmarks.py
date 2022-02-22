import glob
import os
import time
from typing import Any

import streamlit as st

from tokenization_repair.benchmark import run
from tokenization_repair.demo.utils import last_n_k_path

from torch import nn

from trt.utils import common, io
from trt.utils.config import ModelConfig


def show_run_benchmarks(model: nn.Module,
                        model_config: ModelConfig,
                        benchmarks_dir: str,
                        results_dir: str,
                        **kwargs: Any):
    st.write("""
    ## Run the model on various tokenization repair benchmarks
    """)

    tokenization_repair_files = glob.glob(f"{benchmarks_dir}/*/*/corrupt.txt")
    tokenization_repair_output_dirs = {k: os.path.join(results_dir, last_n_k_path(k, k=1)) for k in
                                       tokenization_repair_files}

    benchmark_files = common.natural_sort(tokenization_repair_files)

    select_benchmark = st.selectbox("Select an existing benchmark",
                                    ["-"] + benchmark_files,
                                    index=0,
                                    format_func=last_n_k_path)

    if select_benchmark == "-":
        st.info("Please select a benchmark")
        st.stop()

    benchmark_output_file = f"{model_config.name}-{kwargs.pop('inference_method', 'greedy')}.txt"

    batch_size = st.slider("Batch size for benchmark", min_value=1, max_value=128, value=16)

    is_tokenization_repair = st.checkbox("Tokenization repair", value=True)

    smart_batching = st.checkbox("Smart batching", value=True)

    run_benchmark_button = st.button("Run benchmark")

    output_file = os.path.join(tokenization_repair_output_dirs[select_benchmark], benchmark_output_file)
    st.write(f"*Info: Output file of benchmark will be at `{output_file}`*")

    if run_benchmark_button:
        num_lines = io.line_count(select_benchmark)
        num_batches = max(num_lines // batch_size, 1)
        benchmark_progress = st.progress(0.0)
        eta_str = st.empty()
        current_sequence = st.empty()
        start = time.monotonic()

        i = 0
        for sequences, pred_sequences in run.run_benchmark(input_file=select_benchmark,
                                                           output_file=output_file,
                                                           model=model,
                                                           config=model_config,
                                                           batch_size=batch_size,
                                                           is_tokenization_repair=is_tokenization_repair,
                                                           smart_batching=smart_batching,
                                                           yield_intermediate=True,
                                                           **kwargs):
            sequence_str = f"[{i + 1}/{num_batches}] \n"

            for s, ps in zip(sequences, pred_sequences):
                sequence_str += f"{s} \u2192 {ps}\n"

            current_sequence.code(sequence_str, language=None)

            end = time.monotonic()
            eta_str.write(f"###### *{common.eta_seconds(end - start, i + 1, num_batches)}*")
            benchmark_progress.progress(min(1, (i + 1) / num_batches))
            i += 1

        benchmark_progress.progress(1.0)
        end = time.monotonic()
        eta_str.write(f"###### *Finished in {end - start: .2f} seconds*")
