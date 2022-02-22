import collections
import glob
import os

import altair as alt

import editdistance as ed

import pandas as pd

import streamlit as st

from tokenization_repair.benchmark import evaluate
from tokenization_repair.demo.utils import last_n_k_path

from trt.utils import common


def show_evaluate_benchmarks(benchmarks_dir: str, results_dir: str):
    st.write("""
    ## Evaluate the model on various tokenization repair benchmarks
    """)

    tokenization_repair_files = glob.glob(f"{benchmarks_dir}/*/*/correct.txt")
    evaluation_files = common.natural_sort(tokenization_repair_files)

    select_evaluation = st.selectbox("Select an existing benchmark groundtruth",
                                     ["-"] + evaluation_files,
                                     index=0,
                                     format_func=last_n_k_path)

    if select_evaluation == "-":
        st.info("Please select an benchmark groundtruth to evaluate")
        st.stop()

    evaluation_path = select_evaluation.split("/")

    corrupted_file = os.path.join(benchmarks_dir, evaluation_path[-3], evaluation_path[-2], "corrupt.txt")

    tokenization_repair_results = glob.glob(
        os.path.join(results_dir, evaluation_path[-3], evaluation_path[-2], "*.txt"))
    evaluation_predicted_files = common.natural_sort(tokenization_repair_results)

    select_evaluation_predicted = st.multiselect("Select existing benchmark predictions",
                                                 evaluation_predicted_files,
                                                 format_func=last_n_k_path)

    if len(select_evaluation_predicted) == 0:
        st.info("Please select a predictions file")
        st.stop()

    show_details = st.checkbox("Show detailed results")

    run_evaluation_button = st.button("Run evaluation")

    if run_evaluation_button:
        table_data = collections.defaultdict(dict)
        predicted_files_by_model = {}

        for evaluation_predicted in select_evaluation_predicted:
            model_name = os.path.splitext(os.path.basename(evaluation_predicted))[0]

            seq_acc, mned, med, f1, prec, rec = evaluate.evaluate(groundtruth_file=select_evaluation,
                                                                  predicted_file=evaluation_predicted)

            table_data["Sequence accuracy"].update({model_name: seq_acc})
            table_data["MNED"].update({model_name: mned})
            table_data["MED"].update({model_name: med})
            table_data["F1"].update({model_name: f1})
            table_data["Precision"].update({model_name: prec})
            table_data["Recall"].update({model_name: rec})

            predicted_files_by_model[model_name] = evaluation_predicted

        st.write(f"### Results on benchmark {last_n_k_path(select_evaluation, 3)}:")
        st.dataframe(pd.DataFrame.from_dict(table_data))

        if show_details:
            for key in predicted_files_by_model.keys():
                with st.expander(f"Details for model {key}"):
                    predicted_file = predicted_files_by_model[key]

                    show_details_of_model_predictions(groundtruth_file=select_evaluation,
                                                      predicted_file=predicted_file,
                                                      corrupted_file=corrupted_file)


def show_details_of_model_predictions(groundtruth_file: str,
                                      predicted_file: str,
                                      corrupted_file: str):
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        lengths = []
        equal = []
        edit_distances = []

        errors = []

        for i, (gt, p, c) in enumerate(zip(gtf, pf, cf)):
            eq = gt == p
            equal.append(eq)

            if not eq:
                errors.append((gt, p, c, i))

            lengths.append(len(c))
            edit_distances.append(ed.distance(gt, c))

        df = pd.DataFrame({"sequence length": lengths,
                           "edit distance": edit_distances,
                           "correct": equal})
        chart = alt.Chart(df, title="Correct predictions by sequence length of corrupted input").mark_bar().encode(
            x=alt.X("sequence length:Q", title="sequence length (binned)", bin=alt.Bin(step=5)),
            y="count()",
            color="correct",
            tooltip=["count()", "sequence length", "correct"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        chart = alt.Chart(df,
                          title="Correct predictions by edit distance "
                                "between corrupted input and groundtruth").mark_bar().encode(
            x=alt.X("edit distance:Q", title="Edit distance", bin=alt.Bin(step=1)),
            y="count()",
            color="correct",
            tooltip=["count()", "edit distance", "correct"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    st.write("### Erroneous predictions")
    st.write("")

    detail_string = ""
    for gt, p, c, idx in errors:
        gt = gt.strip()
        p = p.strip()
        c = c.strip()
        detail_string += f"\nSample {idx}:\nInput:\t\t{c}\nGroundtruth:\t{gt}\nPrediction:\t{p}\n"

    st.code(detail_string, language=None)
