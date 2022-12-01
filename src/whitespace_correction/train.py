import argparse
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
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whitespace_correction.model import get_model_from_config
from whitespace_correction.utils import math
from whitespace_correction.utils.config import (
    get_optimizer_from_config,
    get_lr_scheduler_from_config,
    get_loss_from_config, get_data_from_config
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("-c", "--config", type=str, default=None)
    train_group.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument("--overwrite-train-data", type=str, default=None, nargs="+")
    parser.add_argument("--overwrite-val-data", type=str, default=None, nargs="+")
    return parser.parse_args()


# globals used by training and/or evaluation function
step: int = 0  # overall step across epochs
steps_to_fast_forward: int = 0
epoch: int = 0
best_val_loss: float = float("inf")
logger = logging.get_logger("TRAIN")


def prepare_label(
    label: Optional[Dict[str, Any]],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer
) -> Any:
    assert label is not None
    if label["type"] == "sequence_classification":
        num_prefix_tokens = input_tokenizer.num_prefix_tokens()
        num_suffix_tokens = input_tokenizer.num_suffix_tokens()
        return [-1] * num_prefix_tokens + label["labels"] + [-1] * num_suffix_tokens
    else:
        raise ValueError(f"unknown labels type {label['type']}")


def prepare_info(
    info: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer
) -> Dict[str, Any]:
    info_type = info.pop("type")
    if info_type == "empty":
        return {}
    elif info_type == "token_groups":
        groups = info["groups"]
        num_prefix_tokens = input_tokenizer.num_prefix_tokens()
        num_suffix_tokens = input_tokenizer.num_suffix_tokens()
        return {"groups": [1] * num_prefix_tokens + groups + [1] * num_suffix_tokens}
    else:
        raise ValueError(f"unknown info type {info_type}")


def prepare_batch(
    batch: data.Batch,
    model_type: str,
    device: torch.device,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer
) -> Tuple[Dict[str, Any], torch.Tensor]:
    if model_type in {"encoder_with_head"}:
        pad_token_id = input_tokenizer.pad_token_id()
        token_ids = []
        lengths = []
        labels = []
        kwargs = collections.defaultdict(list)
        for item in batch.items:
            token_ids.append(item.tokenization.token_ids)
            lengths.append(len(item.tokenization.token_ids))
            labels.append(prepare_label(item.label, input_tokenizer, output_tokenizer))
            for k, v in prepare_info(item.tokenization.info, input_tokenizer, output_tokenizer):
                kwargs[k].append(v)

        inputs = {
            "token_ids": rnn.pad_sequence(
                [torch.as_tensor(t, dtype=torch.long, device=device) for t in token_ids],
                batch_first=True,
                padding_value=pad_token_id
            ).long(),
            "lengths": lengths,
            **kwargs
        }
        labels = rnn.pad_sequence(
            [torch.as_tensor(t, dtype=torch.long, device=device) for t in labels],
            batch_first=True,
            padding_value=-1
        )

    else:
        raise ValueError(f"unknown model type {model_type}")

    return inputs, labels


def train_one_epoch(
    model: DDP,
    model_type: str,
    info: distributed.DistributedInfo,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    training_steps: int,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    summary_writer: Optional[SummaryWriter],
    checkpoint_dir: Optional[str],
    lr_scheduler: Optional[optim.lr_scheduler.SequentialLR],
    mixed_prec_scaler: amp.GradScaler,
    eval_interval: Union[int, float],
    log_interval: Union[int, float],
) -> None:
    global step
    global steps_to_fast_forward
    global best_val_loss
    global logger
    global epoch

    if isinstance(eval_interval, float):
        eval_interval = math.constrain(int(eval_interval * training_steps), 1, training_steps)
    if isinstance(log_interval, float):
        log_interval = math.constrain(int(log_interval * training_steps), 1, training_steps)

    model = model.train().to(info.device)
    loss_fn = loss_fn.train().to(info.device)

    begin_of_epoch = time.perf_counter()
    start = time.perf_counter()

    mean_loss = tensorboard.AverageTracker("train_loss")
    mean_forward_pass = tensorboard.AverageTracker("train_forward_pass")
    mean_bsz = tensorboard.AverageTracker("train_batch_size")
    mean_seq_length = tensorboard.AverageTracker("train_sequence_length")

    logger.info(f"[rank:{info.rank}] [epoch:{epoch + 1}] training_steps: {training_steps}")

    for batch_num, batch in enumerate(train_loader):
        step += 1
        epoch_step = steps_to_fast_forward + batch_num + 1

        inputs, labels = prepare_batch(
            batch=batch,
            model_type=model_type,
            device=info.device,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )

        optimizer.zero_grad()

        start_forward = time.perf_counter()
        with amp.autocast(enabled=mixed_prec_scaler.is_enabled()):
            outputs, loss_dict = model(**inputs)
            end_forward = time.perf_counter()
            loss = loss_fn(outputs, labels)
            loss = loss + sum(loss_dict.values())

        mixed_prec_scaler.scale(loss).backward()
        mixed_prec_scaler.step(optimizer)
        mixed_prec_scaler.update()

        if info.is_main_process:
            mean_loss.add(loss.detach())
            mean_forward_pass.add((end_forward - start_forward) * 1000)
            # approximation since we expect every process to roughly
            # have the same batch size
            batch_size = labels.shape[0] * info.world_size
            mean_bsz.add(batch_size)
            for item in batch.items:
                mean_seq_length.add(len(item.tokenization.token_ids))

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch_step % eval_interval == 0 and info.is_main_process:
            val_loss = evaluate(
                model=model,
                info=info,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
                val_loader=val_loader,
                loss_fn=loss_fn,
                summary_writer=summary_writer,
            )
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
            io.save_checkpoint(
                checkpoint_path=ckpt_path,
                model=distributed.unwrap_ddp(model),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_scaler=mixed_prec_scaler,
                step=step,
                epoch=epoch,
                epoch_step=epoch_step,
                val_loss=val_loss
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                shutil.copy2(ckpt_path, best_ckpt_path)

            model = model.train()
            loss_fn = loss_fn.train()

        if epoch_step % log_interval == 0 and info.is_main_process:
            if lr_scheduler is not None:
                lr = lr_scheduler.get_last_lr()[0]
                summary_writer.add_scalar("train_lr", lr, step)
                logger.info(f"[{step}, {epoch_step}] train_lr: {lr:.8f}")

            mean_loss.log_tensorboard(summary_writer, step)
            mean_loss.log_info(logger, step)

            mean_bsz.log_tensorboard(summary_writer, step)
            mean_bsz.log_info(logger, step)

            mean_forward_pass.log_tensorboard(summary_writer, step)
            mean_forward_pass.log_info(logger, step)

            mean_seq_length.log_tensorboard(summary_writer, step)
            mean_seq_length.log_info(logger, step)

            summary_writer.add_histogram(
                "train_batch_size_hist",
                mean_bsz.values,
                step
            )

            summary_writer.add_histogram(
                "train_batch_sequence_length_hist",
                mean_seq_length.values,
                step
            )

            end = time.perf_counter()
            logger.info(
                f"[{step}, {epoch_step}] [train_time:{step - log_interval}\u2192{step}] "
                f"{(end - start) / 60:.2f} minutes"
            )
            estimated_steps = train_loader.min_items() // max(mean_bsz.value, 1)
            logger.info(
                f"[{step}, {epoch_step}] [epoch:{epoch + 1}] "
                f"{logging.eta_minutes_message((end - begin_of_epoch) / 60, epoch_step, estimated_steps)}"
            )

            mean_loss.reset()
            mean_bsz.reset()
            mean_forward_pass.reset()
            mean_seq_length.reset()
            start = end


@torch.inference_mode()
def evaluate(
    model: DDP,
    model_type: str,
    info: distributed.DistributedInfo,
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    summary_writer: SummaryWriter
) -> float:
    global step
    global logger

    assert info.is_main_process, "evaluation should be only done on main process"

    mean_loss = tensorboard.AverageTracker("val_loss")

    model = model.eval()
    loss_fn = loss_fn.eval()

    start = time.perf_counter()
    for batch_num, batch in enumerate(val_loader):
        inputs, labels = prepare_batch(
            batch=batch,
            model_type=model_type,
            device=info.device,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer
        )

        outputs, loss_dict = model(**inputs)
        loss = loss_fn(outputs, labels)
        loss = loss + sum(loss_dict.values())

        mean_loss.add(loss.detach())
    end = time.perf_counter()

    mean_loss.log_tensorboard(summary_writer, step)
    mean_loss.log_info(logger, step)

    logger.info(f"[val_time:{step}] {(end - start) / 60:.2f} minutes")
    return mean_loss.value


def train(
    info: distributed.DistributedInfo,
    cfg: Dict[str, Any],
    directories: Dict[str, str],
    resuming: bool
) -> None:
    global logger
    global epoch
    global step
    global steps_to_fast_forward
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

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    input_tokenizer = tokenization.Tokenizer.from_config(cfg["input_tokenizer"])
    if "output_tokenizer" in cfg:
        output_tokenizer = tokenization.Tokenizer.from_config(cfg["output_tokenizer"])
    else:
        output_tokenizer = None

    model = get_model_from_config(
        cfg["model"],
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
    ).to(info.device)

    loss_fn = get_loss_from_config(cfg["train"]["loss"])

    train_loader, _ = get_data_from_config(
        cfg["train"]["data"],
        seed=cfg["seed"],
        epoch=0,
        train_skip=0,
        info=info
    )
    num_batches = 100
    total_batch_size = 0
    logger.info(
        f"Estimating train loader length from average batch size of first {num_batches} batches "
        f"and minimum train loader items."
    )
    idx = 0
    train_loader_iter = iter(train_loader)
    while idx < num_batches:
        batch = next(train_loader_iter)
        total_batch_size += len(batch)
        idx += 1
    avg_batch_size = total_batch_size / num_batches
    training_steps = round(cfg["train"]["num_epochs"] * train_loader.min_items() / avg_batch_size)
    logger.info(f"Got an average batch size of {avg_batch_size:.2f} after {num_batches} batches. "
                f"The train loader contains at least {train_loader.min_items()} items, so the estimated "
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

    mixed_prec_scaler = amp.GradScaler(enabled=cfg["train"].get("mixed_precision", False))

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

        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        steps_to_fast_forward = checkpoint["epoch_step"]

        if info.is_main_process:
            logger.info(
                f"Resuming training from checkpoint {last_checkpoint}\n"
                f"Starting at epoch {epoch + 1} at global step {step} with a best validation loss of {best_val_loss}\n"
                f"Need to fast forward {steps_to_fast_forward} steps"
            )

    ddp_model = DDP(model)

    while epoch < cfg["train"]["num_epochs"]:
        train_loader, val_loader = get_data_from_config(
            cfg["train"]["data"],
            seed=cfg["seed"],
            epoch=epoch,
            train_skip=steps_to_fast_forward,
            info=info
        )

        train_one_epoch(
            model=ddp_model,
            model_type=cfg["model"]["type"],
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            training_steps=training_steps,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            summary_writer=summary_writer,
            info=info,
            checkpoint_dir=directories["checkpoints"],
            lr_scheduler=lr_scheduler,
            mixed_prec_scaler=mixed_prec_scaler,
            eval_interval=cfg["train"]["eval_interval"],
            log_interval=cfg["train"]["log_interval"]
        )

        epoch += 1
        steps_to_fast_forward = 0


def initialize() -> distributed.DistributedInfo:
    assert torch.cuda.device_count() > 0, "need at least one GPU for training, but found none"
    assert dist.is_available(), "distributed package must be available for training"
    assert dist.is_nccl_available(), "nccl backend for distributed training must be available"
    logger = logging.get_logger("TRAIN_INITIALIZATION")
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

    if "SLURM_PROCID" in os.environ and os.environ.get("FORCE_LOCAL", "false") != "true":
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(rank % torch.cuda.device_count())
        local_world_size = torch.cuda.device_count()
        logger.info(
            f"Running on Slurm Cluster: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )
    else:
        assert (
            "RANK" in os.environ
            and "LOCAL_RANK" in os.environ
        ), "could not find RANK and LOCAL_RANK env variables, you probably did not use torchrun to run this script"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        logger.info(
            f"Running using torchrun: master_addr={master_addr}, master_port={master_port}, "
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


def de_initialize() -> None:
    dist.destroy_process_group()


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def setup_experiment_dir(cfg: Dict[str, Any]) -> str:
    experiment_name = re.sub(r"\s", "_", cfg["name"])
    experiment_subdir = f"{experiment_name}@{datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}"
    experiment_dir = os.path.join(cfg["dir"], experiment_subdir)
    os.makedirs(experiment_dir)
    return experiment_dir


def main(args: argparse.Namespace, info: distributed.DistributedInfo) -> None:
    if args.config is not None:
        cfg = configuration.load_config(args.config)
        if info.is_main_process:
            experiment_dir = setup_experiment_dir(cfg["experiment"])
            # save copy of config file to experiment directory
            with open(os.path.join(experiment_dir, "config.yaml"), "w", encoding="utf8") as f:
                f.write(yaml.dump(cfg))
        else:
            experiment_dir = None
        resuming = False
        logger.info(f"Starting experiment at {experiment_dir} with config:\n{yaml.dump(cfg)}")
    else:
        experiment_dir = args.resume
        cfg = configuration.load_config(os.path.join(experiment_dir, "config.yaml"))
        if args.overwrite_train_data is not None:
            logger.info(f"Overriding train data to {args.overwrite_train_data}")
            cfg["train"]["data"] = args.overwrite_train_data
        if args.overwrite_val_data is not None:
            logger.info(f"Overriding val data to {args.overwrite_val_data}")
            cfg["val"]["data"] = args.overwrite_val_data
        resuming = True
        logger.info(f"Resuming from {experiment_dir} with config:\n{yaml.dump(cfg)}")

    directories = {
        "experiment": experiment_dir,
        "checkpoints": os.path.join(experiment_dir, "checkpoints") if experiment_dir is not None else None,
        "tensorboard": os.path.join(experiment_dir, "tensorboard") if experiment_dir is not None else None
    }
    if info.is_main_process:
        os.makedirs(directories["checkpoints"], exist_ok=True)
        os.makedirs(directories["tensorboard"], exist_ok=True)

    # start distributed training
    train(info, cfg, directories, resuming)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args(), initialize())
    de_initialize()
