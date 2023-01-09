from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_correction_utils.api.server import TextCorrectionServer, Error

from whitespace_correction.api.corrector import WhitespaceCorrector


class WhitespaceCorrectionServer(TextCorrectionServer):
    text_corrector_classes = [WhitespaceCorrector]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        @self.server.route(f"{self.base_url}/correct", methods=["POST"])
        def _correct() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            if "task" not in json or "model" not in json:
                return abort(Response("missing task or model in json", status=400))
            try:
                with self.text_corrector(json["task"], json["model"]) as cor:
                    if isinstance(cor, Error):
                        return abort(cor.to_response())
                    assert isinstance(cor, WhitespaceCorrector)
                    corrected = cor.correct_text(json["sequences"])
            except Exception as error:
                return abort(Response(f"request failed with unexpected error: {error}", status=500))

            return jsonify({"sequences": corrected})
