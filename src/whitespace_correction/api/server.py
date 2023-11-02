import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_utils.api.server import TextProcessingServer, Error
from text_utils.api.utils import ProgressIterator
from text_utils import metrics

from whitespace_correction.api.corrector import WhitespaceCorrector


class WhitespaceCorrectionServer(TextProcessingServer):
    text_processor_cls = WhitespaceCorrector

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.batch_size = self.config.get("batch_size", 1)
        if "batch_max_tokens" in self.config:
            self.batch_max_tokens = self.config["batch_max_tokens"]
        else:
            self.batch_max_tokens = None

        @self.server.route(f"{self.base_url}/correct", methods=["POST"])
        def _correct() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            elif "model" not in json:
                return abort(Response("missing model in json", status=400))
            elif "text" not in json:
                return abort(Response("missing text in json", status=400))

            try:
                with self.text_processor(json["model"]) as cor:
                    if isinstance(cor, Error):
                        return abort(cor.to_response())
                    assert isinstance(cor, WhitespaceCorrector)
                    start = time.perf_counter()
                    iter = ProgressIterator(
                        ((t, None) for t in json["text"]),
                        size_fn=lambda e: len(e[0].encode("utf8"))
                    )
                    corrected = list(cor.correct_iter(
                        iter,
                        batch_size=self.batch_size,
                        batch_max_tokens=self.batch_max_tokens,
                    ))
                    end = time.perf_counter()
                    b = iter.total_size
                    s = end - start
            except Exception as error:
                return abort(Response(f"request failed with unexpected error: {error}", status=500))

            return jsonify({
                "text": corrected,
                "runtime": {"b": b, "s": s}
            })

        @self.server.route(f"{self.base_url}/evaluate", methods=["POST"])
        def _evaluate() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            elif "input" not in json:
                return abort(Response("missing input in json", status=400))
            elif "output" not in json:
                return abort(Response("missing output in json", status=400))
            elif "groundtruth" not in json:
                return abort(Response("missing groundtruth in json", status=400))

            try:
                seq_acc = metrics.accuracy(json["output"], json["groundtruth"])
                (f1_micro, *_), _ = metrics.whitespace_correction_f1(
                    json["input"],
                    json["output"],
                    json["groundtruth"],
                    sequence_averaged=False
                )
                (f1_seq_avg, *_), infos = metrics.whitespace_correction_f1(
                    json["input"],
                    json["output"],
                    json["groundtruth"],
                    sequence_averaged=True
                )
            except Exception as error:
                return abort(
                    Response(
                        f"request failed with unexpected error, most likely the "
                        f"request data is malformed: {error}",
                        status=400
                    )
                )

            return jsonify({
                "metrics": [
                    {"name": "Sequence accuracy", "score": seq_acc * 100, "largerIsBetter": True, "precision": 2},
                    {"name": "Micro-averaged F1", "score": f1_micro * 100, "largerIsBetter": True, "precision": 2},
                    {"name": "Sequence-averaged F1", "score": f1_seq_avg * 100, "largerIsBetter": True, "precision": 2},
                ],
                "info": [
                    {"tp": tp, "fp": fp, "fn": fn}
                    for tp, fp, fn in infos
                ]
            })
