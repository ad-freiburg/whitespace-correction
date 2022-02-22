import os

import streamlit as st


def show_info(base_dir: str):
    st.write("""
    ## Info

    This page contains some useful information about this project.

    #### Requirements of the base project from `requirements.txt`
    """)
    with open(os.path.join(base_dir, "requirements.txt"), "r", encoding="utf8") as f:
        st.code(f.read())

    st.write("""
    #### Additional requirements of this project from `tokenization_repair/requirements.txt`
    """)
    with open(os.path.join(base_dir, "tokenization_repair", "requirements.txt"), "r", encoding="utf8") as f:
        st.code(f.read())
