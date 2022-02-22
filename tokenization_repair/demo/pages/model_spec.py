import pandas as pd

import streamlit as st

from torch import nn

from trt.utils import common
from trt.utils.config import Config


def show_model_spec(config: Config, steps: int, model: nn.Module):
    st.write(f"""
    ## Model specification

    - Name: {config.model.name}
    - Type: {config.model.type}
    - Training steps: {steps:,}
    - Batch size: {config.train.batch_size}
    - Batch max tokens: {config.train.batch_max_tokens}
    - Max sequence length: {config.train.max_seq_length}

    ### Detailed specifications
    """)
    with st.expander("Show config file"):
        st.markdown("###### *Info: You can just use this config file "
                    "to reproduce the results of this model*")
        st.code(config, language="yaml")
    with st.expander("Show architecture"):
        st.markdown("###### *Info: The layers inside some submodules may not be in"
                    " the order in which they are called during a forward pass*")
        st.code(model)
    with st.expander("Show number of parameters"):
        if config.model.type == "encoder_with_head":
            st.dataframe(pd.DataFrame.from_dict({
                "Encoder": common.get_num_parameters(model.encoder),
                "Head": common.get_num_parameters(model.head),
                "All": common.get_num_parameters(model),
            }))
        elif config.model.type == "decoder":
            st.dataframe(pd.DataFrame.from_dict({
                "Decoder": common.get_num_parameters(model.decoder),
                "All": common.get_num_parameters(model),
            }))
        else:
            st.dataframe(pd.DataFrame.from_dict({
                "Encoder": common.get_num_parameters(model.encoder),
                "Decoder": common.get_num_parameters(model.decoder),
                "All": common.get_num_parameters(model),
            }))
