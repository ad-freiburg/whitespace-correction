import argparse
import random
import collections
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Union, Any

import torch
import yaml
from text_correction_utils import (
    distributed,
    data,
    configuration,
    io,
    tokenization,
    logging,
    api,
    tensorboard
)
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch import optim
from torch.backends import cudnn  # noqa
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whitespace_correction.model import get_model_from_config
from whitespace_correction.utils import math
from whitespace_correction.utils.config import (
    get_metrics_from_config,
    get_optimizer_from_config,
    get_lr_scheduler_from_config,
    get_loss_from_config,
    get_data_from_config,
    get_tokenizer_from_config
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    return parser.parse_args()


# globals used by training and/or evaluation function
step: int = 0  # overall step across epochs
epoch_step: int = 0
epoch_items: int = 0
epoch: int = 0
best_val_loss: float = float("inf")
logger = logging.get_logger("TRAIN")


def prepare_label(
    label: Optional[Dict[str, Any]],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    max_length: int
) -> Any:
    assert label is not None
    if label["type"] == "classification":
        return label["label"]
    elif label["type"] == "sequence_classification":
        num_prefix_tokens = input_tokenizer.num_prefix_tokens()
        return (
            [-1] * num_prefix_tokens
            + label["labels"]
            + [-1] * (max_length - len(label["labels"]) - num_prefix_tokens)
        )
    else:
        raise ValueError(f"unknown labels type {label['type']}")


def prepare_info(
    info: Dict[str, Any]
) -> Dict[str, Any]:
    info_type = info.pop("type")
    if info_type in {"empty", "token_groups"}:
        return info
    else:
        raise ValueError(f"unknown info type {info_type}")


def prepare_batch(
    batch: data.ItemBatch,
    model_cfg: Dict[str, Any],
    device: torch.device,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer
) -> Tuple[Dict[str, Any], torch.Tensor]:
    assert len(batch.items) > 0, "got empty batch"
    if model_cfg["type"] == "encoder_with_head":
        pad_token_id = input_tokenizer.pad_token_id()
        token_ids = []
        token_lengths = [len(item.tokenization.token_ids) for item in batch]
        max_tokens = max(token_lengths)
        if model_cfg["encoder"]["type"] == "grouping":
            max_groups = max(len(item.tokenization.info["groups"]) for item in batch)
        else:
            max_groups = max_tokens
        labels = []
        kwargs = collections.defaultdict(list)
        for i, item in enumerate(batch):
            token_ids.append(item.tokenization.token_ids + [pad_token_id] * (max_tokens - token_lengths[i]))
            for k, v in prepare_info(item.tokenization.info).items():
                kwargs[k].append(v)

            labels.append(prepare_label(item.label, input_tokenizer, output_tokenizer, max_groups))
        
        # input_tensor = torch.tensor(token_ids, dtype=torch.long).pin_memory().to(device, non_blocking=True)
        inputs = {
            "token_ids": torch.as_tensor(token_ids, dtype=torch.long, device=device),
            "lengths": token_lengths,
            **kwargs
        }
        # labels = torch.tensor(labels, dtype=torch.long).pin_memory().to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    else:
        raise ValueError(f"unknown model type {model_cfg['type']}")

    return inputs, labels


def train_one_epoch(
    model: DDP,
    model_cfg: Dict[str, Any],
    info: distributed.DistributedInfo,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    training_steps: int,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    metric_cfg: Dict[str, Any],
    summary_writer: Optional[SummaryWriter],
    checkpoint_dir: Optional[str],
    lr_scheduler: Optional[optim.lr_scheduler.SequentialLR],
    mixed_prec_scaler: amp.GradScaler,
    mixed_prec_dtype: torch.dtype,
    clip_grad_norm: Optional[float],
    eval_interval: Union[int, float],
    log_interval: Union[int, float],
) -> None:
    global step
    global epoch_step
    global epoch_items
    global best_val_loss
    global logger
    global epoch

    if isinstance(eval_interval, float):
        eval_interval = math.constrain(int(eval_interval * training_steps), 1, training_steps)
    if isinstance(log_interval, float):
        log_interval = math.constrain(int(log_interval * training_steps), 1, training_steps)
    if info.is_main_process:
        logger.info(f"evaluating every {eval_interval:,} steps, logging every {log_interval:,} steps")

    model = model.train().to(info.device)
    loss_fn = loss_fn.train().to(info.device)

    begin_of_epoch = time.perf_counter()
    start = time.perf_counter()

    mean_loss = tensorboard.AverageTracker("train_loss", fmt=".2e")
    mean_forward_pass = tensorboard.AverageTracker("train_forward_pass")
    mean_batch_load = tensorboard.AverageTracker("train_batch_load")
    mean_batch_preparation = tensorboard.AverageTracker("train_batch_preparation")
    mean_bsz = tensorboard.AverageTracker("train_batch_size")
    mean_seq_length = tensorboard.AverageTracker("train_sequence_length")

    if metric_cfg is not None:
        metrics = get_metrics_from_config(metric_cfg, input_tokenizer, output_tokenizer, prefix="train")
    else:
        metrics = []

    logger.info(f"[rank {info.rank}] [epoch {epoch + 1}] training_steps: {training_steps}")

    train_iter = iter(train_loader)
    while True:
        start_batch = time.perf_counter()
        batch = next(train_iter, None)
        end_batch = time.perf_counter()
        if batch is None:
            logger.info(f"[rank {info.rank}] finished epoch {epoch + 1}")
            break

        start_preparation = time.perf_counter()
        inputs, labels = prepare_batch(
            batch=batch,
            model_cfg=model_cfg,
            device=info.device,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )
        end_preparation = time.perf_counter()

        optimizer.zero_grad()

        start_forward = time.perf_counter()
        with amp.autocast(enabled=mixed_prec_scaler.is_enabled(), dtype=mixed_prec_dtype):
            outputs, loss_dict = model(**inputs)
            loss = loss_fn(outputs, labels)
            loss = loss + sum(loss_dict.values())
        end_forward = time.perf_counter()

        mixed_prec_scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            mixed_prec_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        mixed_prec_scaler.step(optimizer)
        mixed_prec_scaler.update()

        if info.is_main_process:
            mean_loss.add(loss.detach())
            mean_forward_pass.add((end_forward - start_forward) * 1000)
            # approximation since we expect every rank to roughly
            # have the same batch size
            batch_size = len(batch) * info.world_size
            mean_bsz.add(batch_size)
            for item in batch:
                mean_seq_length.add(len(item.tokenization.token_ids))
            mean_batch_load.add((end_batch - start_batch) * 1000)
            mean_batch_preparation.add((end_preparation - start_preparation) * 1000)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if hasattr(loss_fn, "step"):
            loss_fn.step()

        step += 1
        epoch_step += 1
        epoch_items += len(batch)
        if epoch_step % eval_interval == 0 and info.is_main_process:
            evaluate_and_checkpoint(
                model=model,
                model_cfg=model_cfg,
                info=info,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
                val_loader=val_loader,
                loss_fn=loss_fn,
                grad_scaler=mixed_prec_scaler,
                mixed_prec_dtype=mixed_prec_dtype,
                metric_cfg=metric_cfg,
                summary_writer=summary_writer,
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            model = model.train()
            loss_fn = loss_fn.train()

        if epoch_step % log_interval == 0 and info.is_main_process:
            if lr_scheduler is not None:
                lr = lr_scheduler.get_last_lr()[0]
                summary_writer.add_scalar("train_lr", lr, step)
                logger.info(f"[step {step}] train_lr: {lr:.8f}")

            mean_loss.log_tensorboard(summary_writer, step)
            mean_loss.log_info(logger, step)

            mean_bsz.log_tensorboard(summary_writer, step)
            mean_bsz.log_info(logger, step)

            mean_forward_pass.log_tensorboard(summary_writer, step)
            mean_forward_pass.log_info(logger, step)

            mean_batch_load.log_tensorboard(summary_writer, step)
            mean_batch_load.log_info(logger, step)

            mean_batch_preparation.log_tensorboard(summary_writer, step)
            mean_batch_preparation.log_info(logger, step)

            mean_seq_length.log_tensorboard(summary_writer, step)
            mean_seq_length.log_info(logger, step)

            for metric in metrics:
                metric.set_values(batch, outputs)
                metric.log_tensorboard(summary_writer, step)
                metric.log_info(logger, step)

            summary_writer.add_histogram(
                "train_batch_size_hist",
                torch.as_tensor(mean_bsz.values),
                step
            )

            summary_writer.add_histogram(
                "train_batch_sequence_length_hist",
                torch.as_tensor(mean_seq_length.values),
                step
            )

            end = time.perf_counter()
            logger.info(
                f"[step {step}] [train_time {step - log_interval}\u2192{step}] "
                f"{(end - start) / 60:.2f} minutes"
            )
            logger.info(
                f"[step {step}] [epoch {epoch + 1}] "
                f"{logging.eta_minutes_message((end - begin_of_epoch) / 60, epoch_step, training_steps)}"
            )

            mean_loss.reset()
            mean_bsz.reset()
            mean_forward_pass.reset()
            mean_batch_load.reset()
            mean_seq_length.reset()
            start = end


@torch.inference_mode()
def evaluate_and_checkpoint(
    model: DDP,
    model_cfg: Dict[str, Any],
    info: distributed.DistributedInfo,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    grad_scaler: amp.GradScaler,
    mixed_prec_dtype: torch.dtype,
    metric_cfg: Dict[str, Any],
    summary_writer: SummaryWriter,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler.SequentialLR],
    checkpoint_dir: str
):
    global step
    global epoch_step
    global epoch_items
    global logger
    global best_val_loss

    assert info.is_main_process, "evaluation should be only done on main process"

    mean_loss = tensorboard.AverageTracker("val_loss", fmt=".2e")

    model = model.eval()
    loss_fn = loss_fn.eval()
    if metric_cfg is not None:
        metrics = get_metrics_from_config(metric_cfg, input_tokenizer, output_tokenizer, prefix="val")
    else:
        metrics = []

    start = time.perf_counter()
    for batch_num, batch in enumerate(val_loader):
        inputs, labels = prepare_batch(
            batch=batch,
            model_cfg=model_cfg,
            device=info.device,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )

        with amp.autocast(enabled=grad_scaler.is_enabled(), dtype=mixed_prec_dtype):
            outputs, loss_dict = model(**inputs)
            loss = loss_fn(outputs, labels)
            loss = loss + sum(loss_dict.values())

        mean_loss.add(loss.detach())
        if batch_num == 0:
            for metric in metrics:
                metric.set_values(batch, outputs)
                metric.log_tensorboard(summary_writer, step)
                metric.log_info(logger, step)

    end = time.perf_counter()

    mean_loss.log_tensorboard(summary_writer, step)
    mean_loss.log_info(logger, step)

    logger.info(f"[step {step}] validation took {(end - start) / 60:.2f} minutes")
    val_loss = mean_loss.value
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    io.save_checkpoint(
        checkpoint_path=ckpt_path,
        model=distributed.unwrap_ddp(model),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        grad_scaler=grad_scaler,
        step=step,
        epoch=epoch,
        epoch_step=epoch_step,
        epoch_items=epoch_items,
        val_loss=val_loss
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        shutil.copy2(ckpt_path, best_ckpt_path)


def train(
    info: distributed.DistributedInfo,
    cfg: Dict[str, Any],
    directories: Dict[str, str],
    resuming: bool
) -> None:
    global logger
    global epoch
    global step
    global epoch_step
    global epoch_items
    global best_val_loss

    if info.is_main_process:
        log_file = os.path.join(directories["experiment"], "logs.txt")
        logging.add_file_log(logger, log_file)

    device_props = api.device_info(info.device)
    logger.info(
        f"[GPU:{info.rank}:{info.local_rank}] {device_props}"
    )

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    torch.use_deterministic_algorithms(False)

    input_tokenizer = get_tokenizer_from_config(cfg["input_tokenizer"])
    if "output_tokenizer" in cfg:
        output_tokenizer = get_tokenizer_from_config(cfg["output_tokenizer"])
    else:
        output_tokenizer = None

    model = get_model_from_config(
        cfg["model"],
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
    ).to(info.device)

    train_loader, val_loader = get_data_from_config(
        cfg["train"]["data"],
        cfg["input_tokenizer"],
        seed=cfg["seed"],
        info=info
    )
    num_batches = 200
    avg_batch_size = tensorboard.AverageTracker("batch_size")
    if info.is_main_process:
        logger.info(
            f"Estimating train loader length from average batch size of first {num_batches} batches "
            f"and minimum train loader items."
        )
    for idx, batch in enumerate(train_loader):
        if idx >= num_batches:
            break
        avg_batch_size.add(len(batch))

    training_steps_per_epoch = int(train_loader.min_items // avg_batch_size.value)
    training_steps = cfg["train"]["num_epochs"] * training_steps_per_epoch
    logger.info(f"Got an average batch size of {avg_batch_size.value:.2f} after {num_batches} batches. "
                f"The train loader contains at least {train_loader.min_items} items, so the estimated "
                f"number of training steps over {cfg['train']['num_epochs']} epochs is {training_steps}.")
    optimizer = get_optimizer_from_config(
        model,
        cfg["train"]["optimizer"]
    )

    if "lr_scheduler" in cfg["train"]:
        lr_scheduler = get_lr_scheduler_from_config(
            optimizer,
            training_steps,
            cfg["train"]["lr_scheduler"]
        )
    else:
        lr_scheduler = None

    loss_fn = get_loss_from_config(training_steps, cfg["train"]["loss"])
    mixed_prec_scaler = amp.GradScaler(enabled=cfg["train"].get("mixed_precision", False))
    mixed_precision_dtype = cfg["train"].get("mixed_precision_dtype", "fp16")
    if mixed_precision_dtype == "fp16":
        mixed_prec_dtype = torch.float16
    elif mixed_precision_dtype == "bfp16":
        mixed_prec_dtype = torch.bfloat16
    else:
        raise ValueError(f"unknown mixed precision type {mixed_precision_dtype}, must fp16 or bfp16")

    if info.is_main_process:
        summary_writer = SummaryWriter(log_dir=directories["tensorboard"])

        logger.info(f"Using model:\n{model}")
        logger.info(f"Model parameters: {api.num_parameters(model)}")

        test_sentence = "This is a test sentence."
        logger.info(f"Testing input tokenizer:\n{input_tokenizer.tokenize(test_sentence).token_ids}")
        if output_tokenizer is not None:
            logger.info(f"Testing output tokenizer:\n{output_tokenizer.tokenize(test_sentence).token_ids}")

        logger.info(f"Type 'tensorboard --logdir {directories['tensorboard']}' "
                    f"to view the training process in Tensorboard")
    else:
        summary_writer = None

    if resuming:
        last_checkpoint = os.path.join(directories["checkpoints"], "checkpoint_last.pt")
        checkpoint = io.load_checkpoint(last_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if checkpoint.get("grad_scaler_state_dict") is not None:
            mixed_prec_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if checkpoint.get("loss_fn_state_dict") is not None:
            loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])

        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        epoch_step = checkpoint["epoch_step"]
        epoch_items = checkpoint["epoch_items"]

        if info.is_main_process:
            logger.info(
                f"Resuming training from checkpoint {last_checkpoint}\n"
                f"Starting at epoch {epoch + 1} at global step {step:,} (epoch step {epoch_step}) with a best validation loss of {best_val_loss:.6f}\n"
                f"Fast forwarding {epoch_items:,} items."
            )

    ddp_model = DDP(model)

    try:
        while epoch < cfg["train"]["num_epochs"]:
            train_loader.set_epoch(epoch)
            train_loader.set_fast_forward(epoch_items)

            train_one_epoch(
                model=ddp_model,
                model_cfg=cfg["model"],
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
                training_steps=training_steps_per_epoch,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                metric_cfg=cfg["train"].get("metrics"),
                summary_writer=summary_writer,
                info=info,
                checkpoint_dir=directories["checkpoints"],
                lr_scheduler=lr_scheduler,
                mixed_prec_scaler=mixed_prec_scaler,
                mixed_prec_dtype=mixed_prec_dtype,
                clip_grad_norm=cfg["train"].get("clip_grad_norm"),
                eval_interval=cfg["train"]["eval_interval"],
                log_interval=cfg["train"]["log_interval"]
            )

            epoch += 1
            epoch_step = 0
            epoch_items = 0
    except KeyboardInterrupt:
        if info.is_main_process:
            logger.info("got termination signal, evaluating and saving on main process before exiting")
    finally:
        if info.is_main_process:
            start = time.perf_counter()
            evaluate_and_checkpoint(
                model=ddp_model,
                model_cfg=cfg["model"],
                info=info,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
                val_loader=val_loader,
                loss_fn=loss_fn,
                grad_scaler=mixed_prec_scaler,
                mixed_prec_dtype=mixed_prec_dtype,
                metric_cfg=cfg["train"].get("metrics"),
                summary_writer=summary_writer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=directories["checkpoints"]
            )
            end = time.perf_counter()
            logger.info(f"final evaluation and checkpointing took {end - start:.2f}s")


def de_initialize() -> None:
    dist.destroy_process_group()


def setup_experiment(dir: str, config_path: str, cfg: Dict[str, Any]):
    os.makedirs(dir, exist_ok=True)
    # save the resolved config to the experiment directory
    with open(os.path.join(dir, "config.yaml"), "w", encoding="utf8") as f:
        f.write(yaml.dump(cfg))
    # make a backup of the raw, unresolved config in the experiment directory
    shutil.copy2(config_path, os.path.join(dir, "config_raw.yaml"))
    os.makedirs(os.path.join(dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(dir, "tensorboard"), exist_ok=True)


def train_local_distributed(
    rank: int,
    world_size: int,
    port: int,
        cfg: Dict[str, Any],
        directories: Dict[str, str],
        resuming: bool
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend=dist.Backend.NCCL,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    assert dist.is_initialized(), "failed to initialize process group"

    info = distributed.DistributedInfo(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size
    )

    train(info, cfg, directories, resuming)

    de_initialize()


def initialize_slurm() -> distributed.DistributedInfo:
    assert torch.cuda.device_count() > 0, "need at least one GPU for training, but found none"
    assert dist.is_available(), "distributed package must be available for training"
    assert dist.is_nccl_available(), "nccl backend for distributed training must be available"
    logger = logging.get_logger("SLURM_INITIALIZATION")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU{'s' * (num_gpus > 1)} "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    assert (
        "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
        and "WORLD_SIZE" in os.environ
    ), f"could not find at least one of MASTER_ADDR, MASTER_PORT and WORLD_SIZE env variables"
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert "SLURM_PROCID" in os.environ, \
        f"distributed training across multiple nodes is only supported with SLURM"
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(rank % torch.cuda.device_count())
    local_world_size = torch.cuda.device_count()
    if rank == 0:
        logger.info(
            f"Running on Slurm Cluster: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )

    dist.init_process_group(
        backend=dist.Backend.NCCL,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    assert dist.is_initialized(), "failed to initialize process group"

    return distributed.DistributedInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size
    )


def train_slurm(args: argparse.Namespace, info: distributed.DistributedInfo) -> None:
    resuming = os.path.exists(args.experiment) and os.path.exists(
        os.path.join(args.experiment, "checkpoints", "checkpoint_last.pt")
    )
    if not resuming:
        assert args.config is not None, "specify config if not resuming an existing experiment"
        cfg = configuration.load_config(args.config)
        if info.is_main_process:
            setup_experiment(args.experiment, args.config, cfg)
            logger.info(f"Starting experiment at {args.experiment} with config:\n{yaml.dump(cfg)}")
    else:
        cfg = configuration.load_config(os.path.join(args.experiment, "config_raw.yaml"))
        if info.is_main_process:
            logger.info(f"Resuming from {args.experiment} with config:\n{yaml.dump(cfg)}")

    directories = {
        "experiment": args.experiment,
        "checkpoints": os.path.join(args.experiment, "checkpoints"),
        "tensorboard": os.path.join(args.experiment, "tensorboard")
    }

    train(info, cfg, directories, resuming)


if __name__ == "__main__":
    args = parse_args()
    resuming = os.path.exists(args.experiment) and os.path.exists(
        os.path.join(args.experiment, "checkpoints", "checkpoint_last.pt")
    )
    if args.local:
        logger = logging.get_logger("LOCAL_INITIALIZATION")
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 0, "need at least one GPU for local training"
        # start local distributed training
        port = int(os.environ.get("MASTER_PORT", random.randint(10000, 60000)))
        if not resuming:
            cfg = configuration.load_config(args.config)
            assert args.config is not None, "specify config if not resuming an existing experiment"
            setup_experiment(args.experiment, args.config, cfg)
            logger.info(f"Starting experiment at {args.experiment} with config:\n{yaml.dump(cfg)}")
        else:
            cfg = configuration.load_config(os.path.join(args.experiment, "config_raw.yaml"))
            logger.info(f"Resuming from {args.experiment} with config:\n{yaml.dump(cfg)}")
        directories = {
            "experiment": args.experiment,
            "checkpoints": os.path.join(args.experiment, "checkpoints"),
            "tensorboard": os.path.join(args.experiment, "tensorboard")
        }
        p_ctx = mp.spawn(
            fn=train_local_distributed,
            nprocs=num_gpus,
            args=(num_gpus, port, cfg, directories, resuming),
            join=True
        )

    else:
        train_slurm(args, initialize_slurm())
        de_initialize()
