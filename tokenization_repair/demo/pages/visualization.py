from typing import Any, Dict, List, Optional

import altair as alt

import numpy as np

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import streamlit as st

from tokenization_repair.demo.pages.interactive import infer
from tokenization_repair.demo.utils import get_attention_type, get_embedding_type

import torch
from torch import nn

from trt.utils import common, hooks, inference, tokenization_repair


def show_visualization(model: nn.Module, model_type: str, **kwargs: Any):
    """
    ## Visualize attention maps and token embeddings
    """

    model_hook = hooks.ModelHook()

    attention_weights_hook = hooks.AttentionWeightsHook()
    model_hook.attach("attention_weights", model, attention_weights_hook, nn.MultiheadAttention)

    # for decoder outputs, attach to the linear layer with name "decoder.out_proj" because that is
    # the final layer which maps to the decoder vocab
    decoder_outputs_hook = hooks.SaveOutputHook()
    model_hook.attach("decoder_outputs", model, decoder_outputs_hook, nn.Linear,
                      layer_name="decoder.out_proj")

    encoder_input_ids_hook = hooks.SaveInputHook()
    model_hook.attach("encoder_input_ids", model, encoder_input_ids_hook, nn.Embedding,
                      layer_name="encoder.embedding.embedding",
                      pre_hook=True)

    encoder_representations_hook = hooks.SaveOutputHook()
    model_hook.attach("encoder_representations", model, encoder_representations_hook,
                      nn.TransformerEncoderLayer)

    decoder_representations_hook = hooks.SaveOutputHook()
    model_hook.attach("decoder_representations", model, decoder_representations_hook,
                      nn.TransformerDecoderLayer)

    inference_results, sequence, is_tokenization_repair = infer(model=model,
                                                                model_type=model_type,
                                                                return_info=True,
                                                                **kwargs)

    if is_tokenization_repair:
        st.write("### Visualize encoder representations")
        to_sequence = st.text_input(f"Write here the correct solution for the sentence from "
                                    f"above without whitespacing errors: \"{sequence}\"")
        if to_sequence:
            if to_sequence.replace(" ", "") != sequence.replace(" ", ""):
                st.error("Correct solution can only differ from the sentence from above in whitespaces.")
                st.stop()

            labels = tokenization_repair.get_whitespace_operations(from_sequence=sequence,
                                                                   to_sequence=to_sequence)
            visualize_encoder_representations(encoder_representations=model_hook["encoder_representations"],
                                              tokens=[f"'{tok}'" for tok in sequence],
                                              labels=labels)

    source_tokens = target_tokens = None

    if model_type == "transformer" or model_type == "decoder":
        visualize_decoder_outputs(inference_results=inference_results,
                                  decoder_outputs=model_hook["decoder_outputs"]["decoder.out_proj"],
                                  model=model,
                                  model_type=model_type)

        target_tokens = [model.decoder.tokenizer.id_to_token(token_id)
                         for token_id in inference_results[0].token_ids]

        if is_tokenization_repair:
            st.write("### Visualize decoder representations")
            to_sequence = st.text_input(f"Write here the correct solution for the sentence from "
                                        f"above without whitespacing errors: \"{sequence}\"", key=123)

            if to_sequence:
                if to_sequence.replace(" ", "") != sequence.replace(" ", ""):
                    st.error("Correct solution can only differ from the sentence from above in whitespaces.")
                    st.stop()

                labels = tokenization_repair.get_whitespace_operations(from_sequence=sequence,
                                                                       to_sequence=to_sequence)

                num_representations = min(len(target_tokens) - 1,
                                          len(list(model_hook["decoder_representations"].values())[-1]),
                                          len(labels) + 1)

                visualize_decoder_representations(decoder_representations=model_hook["decoder_representations"],
                                                  tokens=target_tokens[1:num_representations + 1],
                                                  labels=(labels + ["<eos>"])[:num_representations])

    if "encoder_input_ids" in model_hook.attached_hooks.keys():
        source_token_ids = model_hook["encoder_input_ids"]["encoder.embedding.embedding"][0].squeeze(-1)
        source_tokens = [model.encoder.tokenizer.id_to_token(i) for i in source_token_ids]

    visualize_attention(attention_weights=model_hook["attention_weights"],
                        source_tokens=source_tokens,
                        target_tokens=target_tokens)

    visualize_embeddings(model=model)


def visualize_attention(attention_weights: Dict[str, List[torch.Tensor]],
                        source_tokens: List[str] = None,
                        target_tokens: List[str] = None):
    st.write("### Visualize every attention map that was created while correcting the text above")

    select_attention_map = st.selectbox("Select an attention layer",
                                        options=["-"] + common.natural_sort([
                                            f"{get_attention_type(layer_name)}: {layer_name}"
                                            for layer_name in attention_weights.keys()]))
    st.write("###### *Info: The attention masks are ordered "
             "in the way they get created during a forward pass of the model*")
    st.write("")
    if select_attention_map != "-":
        attention_type, layer_name = select_attention_map.split(":")
        attention_type = attention_type.strip()
        layer_name = layer_name.strip()

        attention = attention_weights[layer_name]

        # decoder self attention
        if attention_type == "Decoder self-attention":
            pad_to = attention[-1].shape[-1]
            attention_maps = [np.pad(attn[0, -1, :], (0, pad_to - attn.shape[-1])) for attn in attention]
            attention_map = np.vstack(attention_maps)
            assert attention_map.ndim == 2 and attention_map.shape[0] == attention_map.shape[1]
            T, _ = attention_map.shape

            x_labels = target_tokens[:T]
            y_labels_left = target_tokens[:T]
            x_label = "decoder input tokens"
            y_label_left = "decoder input tokens"

            output_tokens = target_tokens[1:T + 1]

        # decoder-encoder cross attention
        elif attention_type == "Decoder-encoder cross-attention":
            attention_maps = [attn[0, -1, :] for attn in attention]
            attention_maps.append(torch.zeros(1, 45, dtype=torch.float))
            attention_map = np.vstack(attention_maps)
            T, S = attention_map.shape

            x_labels = source_tokens[:S]
            y_labels_left = target_tokens[:T]
            x_label = "encoder input tokens"
            y_label_left = "decoder input tokens"

            output_tokens = target_tokens[1:T + 1] + ["no output"]

        # encoder self attention
        else:
            assert len(attention) == 1
            attention_map = attention[0][0]
            assert attention_map.ndim == 2 and attention_map.shape[0] == attention_map.shape[1]
            S, _ = attention_map.shape

            x_labels = source_tokens[:S]
            y_labels_left = source_tokens[:S]
            x_label = "encoder input tokens"
            y_label_left = "encoder input tokens"

            output_tokens = None

        chart = attention_chart(name=f"{attention_type}:\n{layer_name}",
                                attention_map=attention_map,
                                x_labels=x_labels,
                                y_labels_left=y_labels_left,
                                x_label=x_label,
                                y_label_left=y_label_left,
                                output_tokens=output_tokens)
        st.altair_chart(chart, use_container_width=True)


def attention_chart(name: str,
                    attention_map: np.ndarray,
                    x_labels: List[str],
                    y_labels_left: List[str],
                    x_label: str,
                    y_label_left: str,
                    output_tokens: Optional[List[str]] = None) -> alt.Chart:
    T, S = attention_map.shape

    x = []
    x_positions = []
    repeated_x_labels = []
    for _ in range(S):
        for i, xl in enumerate(x_labels):
            x.append(f"{xl}{i:04}")
            x_positions.append(i)
            repeated_x_labels.append(xl)

    y_left = []
    y_positions = []
    repeated_y_labels_left = []
    for i, yl in enumerate(y_labels_left):
        for _ in range(T):
            y_left.append(f"{yl}{i:04}")
            y_positions.append(i)
            repeated_y_labels_left.append(yl)

    tooltips = ["x position",
                "y position",
                "x token",
                "y token",
                "weight"]

    df = pd.DataFrame({"x": x,
                       "y_left": y_left,
                       "x position": x_positions,
                       "y position": y_positions,
                       "x token": repeated_x_labels,
                       "y token": repeated_y_labels_left,
                       "weight": attention_map.ravel()})

    if output_tokens is not None:
        repeated_output_tokens = []
        for i, ot in enumerate(output_tokens):
            for _ in range(T):
                repeated_output_tokens.append(ot)

        df["output token"] = repeated_output_tokens

        tooltips.insert(3, "output token")

    x_axis = alt.X("x:N",
                   axis=alt.Axis(title=x_label,
                                 labelExpr="slice(datum.label, 0, -4)",
                                 labelAngle=-45,
                                 labelOverlap=False),
                   sort=None)

    y_axis = alt.Y("y_left:N",
                   axis=alt.Axis(title=y_label_left,
                                 labelExpr="slice(datum.label, 0, -4)",
                                 labelOverlap=False),
                   sort=None)

    chart = alt.Chart(df, title=name).mark_rect().encode(x=x_axis,
                                                         y=y_axis,
                                                         color=alt.Color("weight:Q",
                                                                         scale=alt.Scale(scheme="reds")),
                                                         tooltip=tooltips).interactive().configure_axis(grid=False)

    return chart


def embedding_chart(name: str,
                    embeddings: np.ndarray,
                    labels: List[str]) -> alt.Chart:
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(embeddings)

    tsne = TSNE(perplexity=10, learning_rate=10)
    tsne_output = tsne.fit_transform(embeddings)
    x, y = tsne_output.T
    df = pd.DataFrame({"x": x,
                       "y": y,
                       "cluster": clusters,
                       "token": labels})

    positions = alt.Chart(df, title=f"TSNE {name}"). \
        mark_circle(size=60). \
        encode(x="x", y="y", color="cluster:N", tooltip=["token", "cluster"])

    text = positions.mark_text(
        align="left",
        baseline="middle",
        dx=7
    ).encode(text="token")

    return (positions + text).interactive().configure_axis(grid=False)


def decoder_outputs_chart(name: str,
                          decoder_outputs: np.ndarray,
                          tokens: List[str],
                          top_k: int = 10) -> alt.Chart:
    decoder_dist = torch.softmax(torch.from_numpy(decoder_outputs), dim=0)
    topk_probabilities, topk_indices = torch.topk(decoder_dist, top_k)

    top_k_tokens = [tokens[idx] for idx in topk_indices]

    df = pd.DataFrame({
        "token": top_k_tokens,
        "probability": topk_probabilities.tolist()
    })

    chart = alt.Chart(df, title=name).mark_bar().encode(
        x="token",
        y="probability"
    )
    return chart


def visualize_embeddings(model: nn.Module):
    st.markdown("### Visualize embeddings")

    embeddings_hook = hooks.SaveOutputHook()
    model_hook = hooks.ModelHook()
    model_hook.attach("embeddings", model, embeddings_hook, nn.Embedding)
    learned_embeddings_layers = [f"{get_embedding_type(layer_name)}: {layer_name}"
                                 for layer_name in model_hook["embeddings"].keys()]

    select_embedding = st.selectbox("Select an embedding layer",
                                    options=["-"] + learned_embeddings_layers)
    st.write("###### *Info: If you only see encoder embedding options, "
             "this could mean that the embedding layers of the encoder and decoder are shared in the model*")
    st.write("")

    model_device = next(model.parameters()).device

    if select_embedding != "-":
        embedding_type, layer_name = select_embedding.split(":")
        layer_name = layer_name.strip()

        if embedding_type == "Encoder token embedding":
            embed = model.encoder.embedding.embedding
            input_ids = torch.arange(model.encoder.tokenizer.get_vocab_size(), device=model_device).unsqueeze(
                0).long()
            labels = [repr(model.encoder.tokenizer.id_to_token(token_id))
                      for token_id in range(model.encoder.tokenizer.get_vocab_size())]
        elif embedding_type == "Encoder positional embedding":
            embed = model.encoder.embedding.pos_embedding
            input_ids = torch.arange(model.encoder.config.max_num_embeddings, device=model_device).unsqueeze(0).long()
            labels = [str(i) for i in range(model.encoder.config.max_num_embeddings)]
        elif embedding_type == "Decoder token embedding":
            embed = model.decoder.embedding.embedding
            input_ids = torch.arange(model.decoder.tokenizer.get_vocab_size(), device=model_device).unsqueeze(
                0).long()
            labels = [repr(model.decoder.tokenizer.id_to_token(token_id))
                      for token_id in range(model.decoder.tokenizer.get_vocab_size())]
        else:
            embed = model.decoder.embedding.pos_embedding
            input_ids = torch.arange(model.decoder.config.max_num_embeddings, device=model_device).unsqueeze(0).long()
            labels = [str(i) for i in range(model.decoder.config.max_num_embeddings)]

        _ = embed(input_ids.T)

        embeddings = model_hook["embeddings"]

        embedding = embeddings[layer_name][0].squeeze(1)

        chart = embedding_chart(name=select_embedding,
                                embeddings=embedding,
                                labels=labels)
        st.altair_chart(chart, use_container_width=True)


def visualize_decoder_outputs(inference_results: List[inference.InferenceResult],
                              model: nn.Module,
                              model_type: str,
                              decoder_outputs: torch.Tensor):
    if inference_results is None or model_type == "encoder_with_head":
        return

    target_token_ids = inference_results[0].token_ids

    st.write("### Visualize the distribution over output tokens of the decoder at each time step")

    def _format_fn(position: int) -> str:
        if position == 0:
            return "-"
        return f"Position {position}: " \
               f"\"{model.decoder.tokenizer.decode(target_token_ids[:position])}?\""

    select_visualize_outputs = st.selectbox("Select a time step",
                                            options=list(range(len(decoder_outputs))),
                                            format_func=_format_fn)
    if select_visualize_outputs > 0:
        decoder_output_last = decoder_outputs[select_visualize_outputs - 1][-1, 0, :].squeeze()

        tokens = [model.decoder.tokenizer.id_to_token(token_id)
                  for token_id in range(model.decoder.tokenizer.get_vocab_size())]
        chart = decoder_outputs_chart(name=_format_fn(select_visualize_outputs),
                                      decoder_outputs=decoder_output_last,
                                      tokens=tokens,
                                      top_k=min(10, len(tokens)))
        st.altair_chart(chart, use_container_width=True)


def visualize_decoder_representations(decoder_representations: Dict[str, List[torch.Tensor]],
                                      tokens: List[str],
                                      labels: List[int]):
    alt_charts = []

    for i, (layer_name, outputs) in enumerate(decoder_representations.items()):
        outputs = [r[-1, 0, :] for r in outputs]
        outputs = np.row_stack(outputs)

        tsne = TSNE(perplexity=10, learning_rate=10)
        tsne_output = tsne.fit_transform(outputs)
        x, y = tsne_output.T
        df = pd.DataFrame({"x": x,
                           "y": y,
                           "token": tokens,
                           "class": [f"{label}" for label in labels],
                           "position": [f"{i}" for i in range(len(x))]})

        alt_base = alt.Chart(df, title=f"Decoder layer: {layer_name}").mark_circle(size=60)
        alt_base = alt_base.encode(x="x",
                                   y="y",
                                   tooltip=["class", "token", "position"],
                                   color="class")

        alt_text = alt_base.mark_text(align="left", baseline="middle", dx=7).encode(text="token")

        alt_chart = (alt_base + alt_text).interactive()
        alt_charts.append(alt_chart)

    hstacked_charts = []
    for i in range(0, len(alt_charts), 2):
        hstacked_charts.append(alt.hconcat(*alt_charts[i:i + 2]))

    vstacked_chart = alt.vconcat(*hstacked_charts).configure_axis(grid=False)
    st.altair_chart(vstacked_chart, use_container_width=True)


def visualize_encoder_representations(encoder_representations: Dict[str, List[torch.Tensor]],
                                      tokens: List[str],
                                      labels: List[int]):
    alt_charts = []

    for i, (layer_name, outputs) in enumerate(encoder_representations.items()):
        outputs = outputs[0][1:-1].squeeze(1)

        tsne = TSNE(perplexity=10, learning_rate=10)
        tsne_output = tsne.fit_transform(outputs)
        x, y = tsne_output.T
        df = pd.DataFrame({"x": x,
                           "y": y,
                           "token": tokens,
                           "class": [f"{label}" for label in labels],
                           "position": [f"{i}" for i in range(len(tokens))]})

        alt_base = alt.Chart(df, title=f"Encoder layer: {layer_name}").mark_circle(size=60)
        alt_base = alt_base.encode(x="x",
                                   y="y",
                                   tooltip=["class", "token", "position"],
                                   color="class")

        alt_text = alt_base.mark_text(align="left", baseline="middle", dx=7).encode(text="token")

        alt_chart = (alt_base + alt_text).interactive()
        alt_charts.append(alt_chart)

    hstacked_charts = []
    for i in range(0, len(alt_charts), 2):
        hstacked_charts.append(alt.hconcat(*alt_charts[i:i + 2]))

    vstacked_chart = alt.vconcat(*hstacked_charts).configure_axis(grid=False)
    st.altair_chart(vstacked_chart, use_container_width=True)
