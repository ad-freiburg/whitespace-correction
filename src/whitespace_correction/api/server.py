import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_correction_utils.api.server import TextCorrectionServer, Error
from text_correction_utils.api.utils import ProgressIterator

from whitespace_correction.api.corrector import WhitespaceCorrector


class WhitespaceCorrectionServer(TextCorrectionServer):
    text_corrector_cls = WhitespaceCorrector

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        @self.server.route(f"{self.base_url}/correct", methods=["POST"])
        def _correct() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            if "model" not in json:
                return abort(Response("missing model in json", status=400))
            try:
                with self.text_corrector(json["model"]) as cor:
                    if isinstance(cor, Error):
                        return abort(cor.to_response())
                    assert isinstance(cor, WhitespaceCorrector)
                    start = time.perf_counter()
                    iter = ProgressIterator(
                        ((t, None) for t in json["text"]),
                        size_fn=lambda e: len(e[1].encode("utf8"))
                    )
                    corrected = list(cor.correct_iter(iter))
                    end = time.perf_counter()
                    kb = iter.total_size / 1000
                    s = end - start
                    kbps = kb / s
            except Exception as error:
                return abort(Response(f"request failed with unexpected error: {error}", status=500))

            return jsonify({
                "text": corrected,
                "runtime": {"kb": kb, "s": s, "kbps": kbps}
            })
