import contextlib
import logging
import os
import threading
import time
from typing import Union, Optional, Tuple

import torch.cuda
from flask import Flask, abort, jsonify, request, cli, Response

from ..api import get_available_models, TokenizationRepairer
from ..utils import common

# disable flask startup message and set flask mode to development
cli.show_server_banner = lambda *_: None
os.environ["FLASK_ENV"] = "development"
server = Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = 2 * 1000 * 1000  # 2MB max file size
server_base_url = os.environ.get("BASE_URL", "")
flask_logger = logging.getLogger("werkzeug")
flask_logger.disabled = True
logger = common.get_logger("TRT_SERVER")


class TokenizationRepairers:
    def __init__(self) -> None:
        self.timeout = 1
        self.locks = {}
        self.streams = {}
        self.loaded = False
        self.default_model = ""

    def init(self, timeout: float) -> None:
        num_devices = torch.cuda.device_count()
        logger.info(f"Found {num_devices} GPUs")
        self.timeout = timeout
        for i, model in enumerate(get_available_models()):
            self.locks[model.name] = threading.Lock()
            if num_devices > 0:
                self.streams[model.name] = torch.cuda.Stream()
                tok_rep = TokenizationRepairer.from_pretrained(model.name, i % num_devices)
            else:
                tok_rep = TokenizationRepairer.from_pretrained(model.name, "cpu")
            self.__setattr__(model.name, tok_rep)
            if i == 0:
                self.default_model = model.name
        self.loaded = True

    @contextlib.contextmanager
    def get_repairer(self, name: Optional[str] = None) -> Union[Tuple[str, int], TokenizationRepairer]:
        if name is None:
            name = self.default_model
        # yield either the tokenization repair model or a http status code indicating why this did not work
        if not self.loaded:
            yield "models were not loaded", 500  # internal error, need to load models before using them
        elif not hasattr(self, name):
            yield f"model {name} is not available", 400  # user error, cant request a model that does not exist
        else:
            acquired = self.locks[name].acquire(timeout=self.timeout)
            if not acquired:
                # server capacity is maxed out when acquiring the model did not work within timeout range
                yield f"too many requests, failed to acquire tokenization repairer " \
                      f"within the {self.timeout:.2f}s timeout limit", 503
            else:
                if name in self.streams:
                    with torch.cuda.stream(self.streams[name]):
                        yield self.__getattribute__(name)
                else:
                    yield self.__getattribute__(name)
                self.locks[name].release()


tok_repairers = TokenizationRepairers()


def add_cors_header(response: Response) -> Response:
    response.headers.add("Access-Control-Allow-Origin", os.environ.get("CORS_ORIGIN", "*"))
    return response


@server.route(f"{server_base_url}/models")
def get_models():
    response = jsonify(
        {
            "models": [
                {"name": model.name, "description": model.description}
                for model in get_available_models()
            ],
            "default": tok_repairers.default_model
        }
    )
    return add_cors_header(response)


@server.route(f"{server_base_url}/repair_text", methods=["GET"])
def repair_text():
    start = time.perf_counter()
    text = request.args.get("text")
    if text is None:
        # text is a required field, send bad request as status code
        return abort(add_cors_header(Response("required query parameter \"text\" is missing", status=400)))
    model = request.args.get("model", tok_repairers.default_model)
    with tok_repairers.get_repairer(model) as tok_rep:
        if isinstance(tok_rep, tuple):
            message, status_code = tok_rep
            logger.warning(f"Repairing text aborted with status {status_code}: {message}")
            return abort(add_cors_header(Response(message, status=status_code)))
        else:
            repaired = tok_rep.repair_text(text)
    end = time.perf_counter()
    logger.info(f"Repairing text with {len(text)} chars using model {model} took {end - start:.4f}s")
    response = jsonify(
        {"repaired_text": repaired}
    )
    return add_cors_header(response)


@server.route(f"{server_base_url}/repair_file", methods=["POST"])
def repair_file():
    start = time.perf_counter()
    text = request.form["text"]
    text = [line.strip() for line in text.splitlines()]
    model = request.args.get("model", tok_repairers.default_model)
    with tok_repairers.get_repairer(model) as tok_rep:
        if isinstance(tok_rep, tuple):
            message, status_code = tok_rep
            logger.warning(f"Repairing file aborted with status {status_code}: {message}")
            return abort(add_cors_header(Response(message, status=status_code)))
        else:
            repaired = tok_rep.repair_text(text)
    end = time.perf_counter()
    logger.info(f"Repairing file with {sum(len(l) for l in text)} chars using model {model} took {end - start:.2f}s")
    response = jsonify(
        {
            "repaired_file": repaired
        }
    )
    return add_cors_header(response)


def run_flask_server(host: str, port: int, timeout: float = 10) -> None:
    logger.info("About to start server, loading all available tokenization repair models before starting")
    tok_repairers.init(timeout=timeout)
    logger.info(f"Set the timeout to acquire a model to {timeout:.4f} seconds")
    logger.info("Starting server...")
    server.run(host, port, debug=False, use_reloader=False)
