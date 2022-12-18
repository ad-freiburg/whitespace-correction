import copy
import math
from typing import Dict, Any, List, Tuple, Optional, Callable

import torch
from torch import optim, nn

from whitespace_correction.utils.loss import SeqLoss, FocalLoss

from text_correction_utils import data, io, distributed, tokenization
from text_correction_utils.modules import lr_scheduler
from text_correction_utils.tensorboard import TensorboardMetric, WhitespaceCorrectionMetric


def get_optimizer_from_config(
    model: nn.Module,
    cfg: Dict[str, Any]
) -> optim.Optimizer:
    cfg = copy.deepcopy(cfg)
    opt_type = cfg.pop("type")
    if opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            **cfg
        )
    elif opt_type == "adam":
        return optim.Adam(
            model.parameters(),
            **cfg
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            **cfg
        )
    else:
        raise ValueError(f"Unknown optimizer type {opt_type}")


def get_lr_scheduler_from_config(
    optimizer: optim.Optimizer,
    training_steps: int,
    cfg: Dict[str, Any]
) -> optim.lr_scheduler.SequentialLR:
    cfg = copy.deepcopy(cfg)
    lr_type = cfg.pop("type")
    if lr_type == "linear_with_warmup":
        return lr_scheduler.linear_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "cosine_with_warmup":
        return lr_scheduler.cosine_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "multi_step_with_warmup":
        return lr_scheduler.multi_step_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "constant_with_warmup":
        return lr_scheduler.constant_with_warmup(optimizer, training_steps, **cfg)
    else:
        raise ValueError(f"unknown lr scheduler type {lr_type}")


def _loss_schedule(training_steps: int, schedule_type: str) -> Callable[[int], float]:
    if schedule_type == "linear":
        return lambda step: max(0.0, 1.0 - step / training_steps)
    elif schedule_type == "cosine":
        def _cosine(step: int):
            frac = min(1.0, step / training_steps)
            return 0.5 * (1.0 + math.cos(math.pi * frac))
        return _cosine
    else:
        raise ValueError(f"unknown schedule type {schedule_type}")


def get_loss_from_config(
    training_steps: int,
    cfg: Dict[str, Any],
) -> nn.Module:
    cfg = copy.deepcopy(cfg)
    loss_type = cfg.pop("type")
    if loss_type == "sequence_cross_entropy":
        cfg["type"] = "cross_entropy"
        loss = get_loss_from_config(training_steps, cfg)
        return SeqLoss(loss=loss)

    elif loss_type == "cross_entropy":
        weight = cfg.get("weights", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.CrossEntropyLoss(ignore_index=cfg.get("ignore_index", -1), weight=weight)
        return loss

    elif loss_type == "binary_cross_entropy":
        weight = cfg.get("weight", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.BCELoss(weight=weight)
        return loss

    elif loss_type == "focal":
        weight = cfg.get("weight", None)
        if "gamma_schedule" in cfg:
            schedule = _loss_schedule(training_steps, cfg["gamma_schedule"])
        else:
            schedule = None
        loss = FocalLoss(
            weight,
            gamma=cfg.get("gamma", 2.),
            ignore_index=cfg.get("ignore_index", -1),
            gamma_schedule=schedule
        )
        return loss

    elif loss_type == "sequence_focal":
        cfg["type"] = "focal"
        loss = get_loss_from_config(training_steps, cfg)
        return SeqLoss(loss=loss)

    else:
        raise ValueError(f"unknown loss type {loss_type}")


def get_tokenizer_config(cfg: Dict[str, Any]) -> tokenization.TokenizerConfig:
    if "language" in cfg:
        language = tokenization.LanguageConfig(**cfg.pop("language"))
    else:
        language = None

    return tokenization.TokenizerConfig(
        **cfg,
        language=language
    )


def get_tokenizer_from_config(cfg: Dict[str, Any]) -> tokenization.Tokenizer:
    return tokenization.Tokenizer.from_config(get_tokenizer_config(cfg))


def _parse_data_sources(sources: List[Dict[str, Any]]) -> Tuple[List[str], Optional[List[str]]]:
    source_paths = []
    source_languages = []
    for src in sources:
        src_type = src.pop("type")
        if src_type == "file":
            lang = src.get("language")
            path = src["path"]
            source_paths.append(path)
            source_languages.append(lang)
        elif src_type == "file_glob":
            lang = src.get("language")
            for path in io.glob_safe(src["glob"]):
                source_paths.append(path)
                source_languages.append(lang)
        else:
            raise ValueError(f"unknown source type {src_type}")

    if all(lang is None for lang in source_languages):
        return source_paths, None
    else:
        return source_paths, [lang if lang is not None else "[unk]" for lang in source_languages]


def get_data_from_config(
    cfg: Dict[str, Any],
    input_tokenizer_cfg: Dict[str, Any],
    seed: int,
    info: distributed.DistributedInfo
) -> Tuple[data.DataLoader, data.DataLoader]:
    cfg = copy.deepcopy(cfg)
    val_cfg = cfg.pop("val")
    if isinstance(val_cfg, int):
        val_limit = val_cfg
        val_sources = val_languages = None
    elif isinstance(val_cfg, list):
        val_limit = None
        val_sources, val_languages = _parse_data_sources(val_cfg)
    else:
        raise ValueError("val data must either be an integer or a list of data sources")

    train_sources, train_languages = _parse_data_sources(cfg.pop("sources"))

    pipeline = cfg.pop("pipeline")
    pipeline_config = data.PipelineConfig(
        preprocessing=pipeline.get("preprocessing", []),
        labeling=pipeline["labeling"],
    )
    tokenizer_config = get_tokenizer_config(input_tokenizer_cfg)
    train_loader = data.DataLoader.from_files(
        train_sources,
        pipeline_config,
        tokenizer_config,
        train_languages,
        seed=seed,
        skip=val_limit if val_limit is not None else 0,
        distributed=(info.rank, info.world_size),
        **cfg
    )

    if val_limit is not None:
        val_loader = data.DataLoader.from_files(
            train_sources,
            pipeline_config,
            tokenizer_config,
            train_languages,
            seed=seed,
            limit=val_limit,
            distributed=None,
            **cfg
        )
    else:
        val_loader = data.DataLoader.from_files(
            val_sources,
            pipeline_config,
            tokenizer_config,
            val_languages,
            seed=seed,
            distributed=None,
            **cfg
        )
    return train_loader, val_loader


def get_metrics_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    prefix: str
) -> List[TensorboardMetric]:
    metrics = []
    for metric_type, metric_opts in cfg.items():
        if metric_type == "whitespace_correction":
            metric = WhitespaceCorrectionMetric(f"{prefix}_whitespace_correction", input_tokenizer, **metric_opts)
        else:
            raise ValueError(f"unknown metric type {metric_type}")
        metrics.append(metric)
    return metrics
