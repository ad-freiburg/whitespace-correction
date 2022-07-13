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

from whitespace_correction import version
from whitespace_correction.api import ModelInfo, WhitespaceCorrector, get_available_models, utils
from whitespace_correction.utils import common, metrics, nlp

# disable flask startup message and set flask mode to development
cli.show_server_banner = lambda *_: None
os.environ["FLASK_ENV"] = "development"
server = Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = 1 * 1000 * 1000  # 1MB max file size
server_base_url = os.environ.get("BASE_URL", "")
flask_logger = logging.getLogger("werkzeug")
flask_logger.disabled = True
logger = common.get_logger("WSC_SERVER")


class WhitespaceCorrectors:
    def __init__(self) -> None:
        self.timeout = 1.0
        self.locks: Dict[str, threading.Lock] = {}
        self.streams: Dict[str, torch.cuda.Stream] = {}
        self.loaded = False
        self.default_model = ""
        self.precision = "fp32"

    def init(self, models: List[str], timeout: float, precision: str) -> None:
        num_devices = torch.cuda.device_count()
        logger.info(f"Found {num_devices} GPUs")
        self.timeout = timeout
        self.precision = precision
        for i, model in enumerate(models):
            self.locks[model] = threading.Lock()
            if num_devices > 0:
                self.streams[model] = torch.cuda.Stream()
                ws_cor = WhitespaceCorrector.from_pretrained(model, i % num_devices)
            else:
                ws_cor = WhitespaceCorrector.from_pretrained(model, "cpu")
            ws_cor.set_precision(precision)
            self.__setattr__(model, ws_cor)
            if i == 0:
                self.default_model = model
        self.loaded = True

    @property
    def available_models(self) -> List[ModelInfo]:
        all_available_models = get_available_models()
        return [model for model in all_available_models if model.name in self.locks]

    @contextlib.contextmanager
    def get_corrector(self, name: Optional[str] = None) -> Generator:
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


ws_correctors = WhitespaceCorrectors()


@server.after_request
def after_request(response: Response) -> Response:
    response.headers.add("Access-Control-Allow-Origin", os.environ.get("CORS_ORIGIN", "*"))
    response.headers.add("Access-Control-Allow-Private-Network", "true")
    return response


@server.route(f"{server_base_url}/models")
def get_models() -> Response:
    response = jsonify(
        {
            "models": [
                {"name": model.name, "description": model.description}
                for model in ws_correctors.available_models
            ],
            "default": ws_correctors.default_model
        }
    )
    return response


@server.route(f"{server_base_url}/info")
def get_info() -> Response:
    response = jsonify(
        {
            "gpu": [utils.get_gpu_info(i) for i in range(torch.cuda.device_count())],
            "cpu": utils.get_cpu_info(),
            "timeout": ws_correctors.timeout,
            "precision": ws_correctors.precision,
            "version": version.__version__
        }
    )
    return response


@server.route(f"{server_base_url}/evaluate", methods=["POST"])
def evaluate() -> Response:
    start = time.perf_counter()
    ipt = request.form.get("input")
    opt = request.form.get("output")
    gt = request.form.get("groundtruth")
    if ipt is None or opt is None or gt is None:
        return abort(Response(
            "request missing one or more of the required fields 'input', 'output', and 'groundtruth'", status=400
        ))

    # clean and split, remove empty final line if it exists
    ipt = [nlp.clean_sequence(i) for i in ipt.split("\n")]
    opt = [nlp.clean_sequence(o) for o in opt.split("\n")]
    gt = [nlp.clean_sequence(g) for g in gt.split("\n")]
    if len(ipt) > 1 and ipt[-1] == "":
        ipt = ipt[:-1]
    if len(opt) > 1 and opt[-1] == "":
        opt = opt[:-1]
    if len(gt) > 1 and gt[-1] == "":
        gt = gt[:-1]

    if not len(ipt) == len(opt) and len(opt) == len(gt):
        return abort(Response(
            f"expected the same number of inputs, outputs, and groundtruths, but got {(len(ipt), len(opt), len(gt))}",
            status=400
        ))

    try:
        _, _, _, f1, prec, rec = metrics.whitespace_correction_f1_prec_rec(opt, gt, ipt)
    except AssertionError:
        return abort(Response(
            "evaluation failed, make sure all inputs, outputs, and groundtruths only differ in whitespaces",
            status=400
        ))

    end = time.perf_counter()
    logger.info(f"Evaluating text with {sum(len(i) for i in ipt)} chars took {end - start:.2f}s")

    return jsonify({
        "precision": prec,
        "recall": rec,
        "f1": f1
    })


@server.route(f"{server_base_url}/correct_text", methods=["POST"])
def repair_text() -> Response:
    start = time.perf_counter()
    text = request.form.get("text")
    if text is None:
        return abort(Response("request missing 'text' field in form data", status=400))
    text = [line.strip() for line in text.splitlines()]
    model = request.args.get("model", ws_correctors.default_model)
    with ws_correctors.get_corrector(model) as ws_cor:
        if isinstance(ws_cor, tuple):
            message, status_code = ws_cor
            logger.warning(f"Correcting text aborted with status {status_code}: {message}")
            return abort(Response(message, status=status_code))
        else:
            corrected = ws_cor.correct_text(text)
    end = time.perf_counter()
    runtime = end - start
    logger.info(f"Correcting text with {sum(len(l) for l in text)} chars using model {model} took {runtime:.2f}s")
    response = jsonify(
        {
            "corrected_text": corrected,
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

    ws_correctors.init(models=models, timeout=timeout, precision=precision)

    logger.info(f"Starting server on {host}:{port}...")
    server.run(host, port, debug=False, use_reloader=False)
