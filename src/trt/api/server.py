import contextlib
import json
import logging
import os
import pprint
import threading
import time
from typing import Dict, Generator, List, Optional

from flask import Flask, Response, abort, cli, jsonify, request

import torch.cuda

from trt.api import TokenizationRepairer, get_available_models, ModelInfo
from trt.utils import common

# disable flask startup message and set flask mode to development
cli.show_server_banner = lambda *_: None
os.environ["FLASK_ENV"] = "development"
server = Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = 1 * 1000 * 1000  # 1MB max file size
server_base_url = os.environ.get("BASE_URL", "")
flask_logger = logging.getLogger("werkzeug")
flask_logger.disabled = True
logger = common.get_logger("TRT_SERVER")


class TokenizationRepairers:
    def __init__(self) -> None:
        self.timeout = 1.0
        self.locks: Dict[str, threading.Lock] = {}
        self.streams: Dict[str, torch.cuda.Stream] = {}
        self.loaded = False
        self.default_model = ""

    def init(self, models: List[str], timeout: float, precision: str) -> None:
        num_devices = torch.cuda.device_count()
        logger.info(f"Found {num_devices} GPUs")
        self.timeout = timeout
        for i, model in enumerate(models):
            self.locks[model] = threading.Lock()
            if num_devices > 0:
                self.streams[model] = torch.cuda.Stream()
                tok_rep = TokenizationRepairer.from_pretrained(model, i % num_devices)
            else:
                tok_rep = TokenizationRepairer.from_pretrained(model, "cpu")
            tok_rep.set_precision(precision)
            self.__setattr__(model, tok_rep)
            if i == 0:
                self.default_model = model
        self.loaded = True

    @property
    def available_models(self) -> List[ModelInfo]:
        all_available_models = get_available_models()
        return [model for model in all_available_models if model.name in self.locks]

    @contextlib.contextmanager
    def get_repairer(self, name: Optional[str] = None) -> Generator:
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
                yield f"server is overloaded with too many requests, failed to reserve tokenization repair model " \
                      f"within the {self.timeout:.2f}s timeout limit", 503
            else:
                if name in self.streams:
                    with torch.cuda.stream(self.streams[name]):
                        yield self.__getattribute__(name)
                else:
                    yield self.__getattribute__(name)
                self.locks[name].release()


tok_repairers = TokenizationRepairers()


@server.after_request
def after_request(response: Response) -> Response:
    response.headers.add("Access-Control-Allow-Origin", os.environ.get("CORS_ORIGIN", "*"))
    return response


@server.route(f"{server_base_url}/models")
def get_models() -> Response:
    response = jsonify(
        {
            "models": [
                {"name": model.name, "description": model.description}
                for model in tok_repairers.available_models
            ],
            "default": tok_repairers.default_model
        }
    )
    return response


@server.route(f"{server_base_url}/repair_text", methods=["GET"])
def repair_text() -> Response:
    start = time.perf_counter()
    text = request.args.get("text")
    if text is None:
        # text is a required field, send bad request as status code
        return abort(Response("required query parameter \"text\" is missing", status=400))
    model = request.args.get("model", tok_repairers.default_model)
    with tok_repairers.get_repairer(model) as tok_rep:
        if isinstance(tok_rep, tuple):
            message, status_code = tok_rep
            logger.warning(f"Repairing text aborted with status {status_code}: {message}")
            return abort(Response(message, status=status_code))
        else:
            repaired = tok_rep.repair_text(text)
    end = time.perf_counter()
    runtime = end - start
    logger.info(f"Repairing text with {len(text)} chars using model {model} took {runtime:.4f}s")
    response = jsonify(
        {
            "repaired_text": repaired,
            "runtime": {
                "total": runtime,
                "cps": len(text) / runtime
            }
        }
    )
    return response


@server.route(f"{server_base_url}/repair_file", methods=["POST"])
def repair_file() -> Response:
    start = time.perf_counter()
    text = request.form.get("text")
    if text is None:
        return abort(Response("request missing 'text' field in form data", status=400))
    text = [line.strip() for line in text.splitlines()]
    model = request.args.get("model", tok_repairers.default_model)
    with tok_repairers.get_repairer(model) as tok_rep:
        if isinstance(tok_rep, tuple):
            message, status_code = tok_rep
            logger.warning(f"Repairing file aborted with status {status_code}: {message}")
            return abort(Response(message, status=status_code))
        else:
            repaired = tok_rep.repair_text(text)
    end = time.perf_counter()
    runtime = end - start
    logger.info(f"Repairing file with {sum(len(l) for l in text)} chars using model {model} took {runtime:.2f}s")
    response = jsonify(
        {
            "repaired_file": repaired,
            "runtime": {
                "total": runtime,
                "cps": sum(len(line) for line in text) / runtime
            }
        }
    )
    return response


def run_flask_server(config_path: str) -> None:
    if not os.path.exists(config_path):
        raise RuntimeError(f"server config file {config_path} does not exist")

    with open(config_path, "r", encoding="utf8") as cfg_file:
        config = json.load(cfg_file)

    assert "host" in config and "port" in config, "'host' and 'port' are required keys in the server config file"
    host = config["host"]
    port = config["port"]

    timeout = config.get("timeout", 10)
    models = config.get("models", [model.name for model in get_available_models()])
    precision = config.get("precision", "fp32")
    logger.info(f"Loaded config for server:\n{pprint.pformat(config)}")

    tok_repairers.init(models=models, timeout=timeout, precision=precision)

    logger.info(f"Starting server on {host}:{port}...")
    server.run(host, port, debug=False, use_reloader=False)
