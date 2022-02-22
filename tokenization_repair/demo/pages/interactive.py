import dataclasses
import time
from typing import Any

import streamlit as st

from tokenization_repair.demo.utils import format_inference_result, get_example_sequences

from torch import nn

from trt.utils import tokenization_repair, inference


def infer(model: nn.Module,
          model_type: str,
          return_info: bool = False,
          **kwargs: Any):
    inference_input = st.text_input("Write something not tokenized correctly")
    st.write("**or**")
    select_sequence = st.selectbox("Choose an example sequence",
                                   ["-"] + get_example_sequences(),
                                   index=0)

    is_tokenization_repair = st.checkbox("Tokenization repair", value=True)

    if inference_input == "" and select_sequence == "-":
        st.info("Please input some wrong text first or choose an example sequence")
        st.stop()

    elif inference_input != "" and select_sequence != "-":
        st.info("Please either input or own text or choose an example, but not both "
                "at the same time")
        st.stop()

    sequence = select_sequence if select_sequence != "-" else inference_input

    time_placeholder = st.empty()
    pred_placeholder = st.empty()

    start = time.monotonic()

    if model_type == "transformer":
        inference_results = model.inference(sequences=[sequence],
                                            **kwargs)[0]

        if isinstance(inference_results[0], list):
            inference_results = [ir for ir in inference_results[0]]

        predicted_sequences = [model.decoder.tokenizer.decode(ir.token_ids)
                               for ir in inference_results]

        predicted_sequences = [format_inference_result(sequence=sequence,
                                                       predicted_sequence=predicted_sequence,
                                                       transform_result_fn=tokenization_repair.repair_whitespace
                                                       if is_tokenization_repair else None)
                               for predicted_sequence in predicted_sequences]

        end = time.monotonic()

        additional_data = [dataclasses.asdict(r) for r in inference_results]

    elif model_type == "encoder_with_head":
        inference_result = model.inference(sequences=[sequence])[0]
        assert isinstance(inference_result, inference.SequenceClassificationInferenceResult)

        if is_tokenization_repair:
            predicted_sequence = tokenization_repair.repair_whitespace(
                sequence=sequence,
                repair_sequence=inference_result.predictions[1:-1]
            )
        else:
            predicted_sequence = f"{inference_result.predictions}"

        predicted_sequences = [predicted_sequence]

        end = time.monotonic()

        additional_data = dataclasses.asdict(inference_result)

    else:
        raise NotImplementedError()

    time_placeholder.write(f"###### *Corrected input in {end - start:.2f} seconds*")

    pred_placeholder.code("\n".join(predicted_sequences), language=None)

    with st.expander("Additional information"):
        st.json(additional_data)

    if return_info:
        return inference_results, sequence, is_tokenization_repair


def show_interactive(model: nn.Module, model_type: str, **kwargs: Any):
    st.write("""
            ## Interactive Demo

            This demo lets you play around with trained
            tokenization repair models.

            ---
            """)

    infer(model=model,
          model_type=model_type,
          **kwargs)
