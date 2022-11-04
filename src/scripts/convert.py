import argparse
import logging
import os
import time
from typing import Callable

import torch
from torch import nn

from whitespace_correction.api.utils import get_tensorrt_cache_dir
from whitespace_correction import WhitespaceCorrector
from whitespace_correction.model.transformer import TransformerEncoderModelWithHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-f", "--format", choices=["torchscript", "onnx", "tensorrt"], required=True)
    parser.add_argument("-o", "--out-file", type=str, default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--allow-pytorch-fallback", action="store_true")
    return parser.parse_args()


class ExportableModel(nn.Module):
    def __init__(self, model: TransformerEncoderModelWithHead) -> None:
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids:  [B, S]
        enc, _ = self.model.encoder(torch.transpose(token_ids, 0, 1))
        # enc: [S, B, H]
        output = self.model.head(enc)
        # output: [S, B, C]
        # return: [B, S, C]
        return torch.transpose(output, 0, 1)


def bench(name: str, placeholder_input: torch.Tensor, model: Callable) -> None:
    print("-" * 80)
    print(f"Benchmarking {name} model")
    print("-" * 80)
    num_iter = 20
    seq_lengths = [512, 256, 128, 64]
    for seq_len in seq_lengths:
        sub_input = placeholder_input[:, :seq_len]
        print(f"Warming up")
        for i in range(num_iter):
            _ = model(sub_input)
        start = time.perf_counter()
        for i in range(num_iter):
            _ = model(sub_input)
        end = time.perf_counter()
        print(
            f"Avg inference speed for input with shape {sub_input.shape}: {1000 * (end - start) / num_iter:.2f}ms"
        )


def export(args: argparse.Namespace) -> None:
    logging.disable(logging.ERROR)

    wr = WhitespaceCorrector.from_experiment(args.experiment, "cuda:0")
    assert isinstance(wr.model, TransformerEncoderModelWithHead), \
        "onnx export is only supported for encoder only models for now"

    placeholder_input = torch.randint(
        0,
        wr.model.encoder.tokenizer.get_vocab_size(),
        (16, wr.model.encoder.embedding.max_num_embeddings),
        dtype=torch.int32,
        device=wr.device
    )

    model = ExportableModel(wr.model).eval()
    if args.bench:
        bench("PyTorch", placeholder_input, model)

    if args.format == "torchscript":
        ts_model = torch.jit.optimize_for_inference(torch.jit.script(model))
        torch.jit.save(
            ts_model, args.out_file if args.out_file is not None else os.path.join(args.experiment, "model.ts")
        )
        if args.bench:
            bench("TorchScript", placeholder_input, ts_model)
    elif args.format == "onnx" or args.format == "tensorrt":
        out_path = args.out_file if args.out_file is not None else os.path.join(args.experiment, "model.onnx")
        torch.onnx.export(
            model,
            placeholder_input,
            out_path,
            export_params=True,
            do_constant_folding=True,
            input_names=["token_ids"],
            output_names=["logits"],
            dynamic_axes={
                "token_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=args.opset
        )
        import onnx
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = len(os.sched_getaffinity(0))
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.enable_cpu_mem_arena = 1

        providers = [
            ("CUDAExecutionProvider", {
                "device_id": wr.device.index or 0
            }),
            "CPUExecutionProvider"
        ]
        if args.format == "tensorrt":
            providers.insert(0, ("TensorrtExecutionProvider", {
                "device_id": wr.device.index or 0,
                "trt_max_workspace_size": 2147483648,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": get_tensorrt_cache_dir()
            }))
        else:
            sess_options.optimized_model_filepath = out_path + ".optimized"

        ort_sess = ort.InferenceSession(
            out_path,
            sess_options,
            providers
        )

        def onnx_model(ipt: torch.Tensor):
            _ = ort_sess.run(["logits"], {"token_ids": ipt.numpy()})

        if args.bench:
            bench("onnx", placeholder_input.cpu(), onnx_model)


if __name__ == "__main__":
    export(parse_args())
