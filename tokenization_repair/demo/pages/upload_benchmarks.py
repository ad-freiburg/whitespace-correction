import os
from io import StringIO

import streamlit as st


def show_upload_benchmarks(benchmarks_dir: str):
    st.write("""
    ## Upload your own tokenization repair benchmarks
    
    The benchmark should be provided as plain text file where each line is one sequence that should be repaired. The
    corresponding groundtruth file should be also a plain text file with the same number of lines as the benchmark, 
    where each line is the groundtruth sequence without tokenization errors.
    
    After you have uploaded your benchmark here, you can run the tokenization repair models on it on the page 
    **Run benchmarks** and then evaluate the results on the page **Evaluate on benchmarks**.
    """)

    st.write("### Upload benchmark")

    upload_benchmark = st.file_uploader("Upload a benchmark file")

    if not upload_benchmark:
        st.info("Please upload a benchmark file")
        st.stop()

    benchmark_data = StringIO(upload_benchmark.getvalue().decode("utf8")).readlines()

    st.write("###### Benchmark preview (first 20 lines):")
    st.json(benchmark_data[:20])

    st.write("### Upload groundtruth")

    upload_groundtruth = st.file_uploader("Upload the corresponding groundtruth for the benchmark")

    if not upload_groundtruth:
        st.info("Please upload the groundtruth file for the benchmark")
        st.stop()

    groundtruth_data = StringIO(upload_groundtruth.getvalue().decode("utf8")).readlines()

    st.write("###### Groundtruth preview (first 20 lines):")
    st.json(groundtruth_data[:20])

    if len(groundtruth_data) != len(benchmark_data):
        st.error(f"Expected benchmark and groundtruth to have the same number of lines, but got {len(benchmark_data)} "
                 f"for benchmark and {len(groundtruth_data)} for groundtruth.")
        st.stop()

    benchmark_name = st.text_input("Name of the benchmark")

    new_benchmark_dir = os.path.join(benchmarks_dir, benchmark_name)

    if benchmark_name == "" or " " in benchmark_name:
        st.info("Please enter a valid benchmark name without whitespaces")
        st.stop()
    elif os.path.exists(new_benchmark_dir):
        st.error(f"Benchmark with name {benchmark_name} already exits. Please choose a different name.")
        st.stop()

    os.makedirs(os.path.join(new_benchmark_dir, "test"))
    with open(os.path.join(new_benchmark_dir, "test", "corrupt.txt"), "w", encoding="utf8") as f:
        f.writelines(benchmark_data)
    with open(os.path.join(new_benchmark_dir, "test", "correct.txt"), "w", encoding="utf8") as f:
        f.writelines(groundtruth_data)

    st.write(f"*Info: Your benchmark is now available as `{benchmark_name}/test/corrupt.txt`, "
             f"the corresponding groundtruth as `{benchmark_name}/test/correct.txt`*")
