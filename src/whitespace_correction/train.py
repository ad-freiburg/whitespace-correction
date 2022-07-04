import argparse
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

import GPUtil

import numpy as np

import tokenizers

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch import optim
from torch.backends import cudnn  # noqa
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whitespace_correction.model import transformer
from whitespace_correction.utils import common, config, constants, data, io, loss, lr_schedule, metrics
from whitespace_correction.utils.lr_schedule import LR_SCHEDULER_TYPE
from whitespace_correction.utils.optimizer import get_optimizer_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable parallelism for tokenizers explicitly
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("-c", "--config", type=str, default=None)
    train_group.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument("--overwrite-train-data", type=str, default=None)
    parser.add_argument("--overwrite-val-data", type=str, default=None)
    return parser.parse_args()


# globals used by training and/or evaluation function
step: int = 0
steps_to_fast_forward: int = 0
epoch: int = 0
best_val_loss: float = float("inf")
logger = common.get_logger("TRAIN")


def prepare_batch(batch: Dict[str, torch.Tensor],
                  device: torch.device,
                  model_type: str) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    if model_type == "encoder_with_head":
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        inputs: Tuple[torch.Tensor, ...] = (input_ids.T,)
        labels = batch["labels"].to(device, non_blocking=True)

    elif model_type == "decoder":
        target_input_ids = batch["input_ids"][:, :-1].to(device, non_blocking=True)

        inputs = (target_input_ids.T,)
        labels = batch["input_ids"][:, 1:].to(device, non_blocking=True)

    else:
        # standard transformer
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_input_ids = batch["target_input_ids"][:, :-1].to(device, non_blocking=True)

        inputs = (input_ids.T, target_input_ids.T)
        labels = batch["target_input_ids"][:, 1:].to(device, non_blocking=True)

    return inputs, labels


def train_one_epoch(model: DDP,
                    encoder_tokenizer: tokenizers.Tokenizer,
                    decoder_tokenizer: tokenizers.Tokenizer,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    summary_writer: SummaryWriter,
                    device: torch.device,
                    checkpoint_dir: str,
                    lr_scheduler: Optional[LR_SCHEDULER_TYPE],
                    mixed_prec_scaler: amp.GradScaler,
                    eval_interval: int,
                    log_interval: int,
                    text_metrics: List[config.MetricConfig],
                    keep_last_n_checkpoints: int,
                    model_name: str,
                    model_type: str) -> None:
    global step
    global steps_to_fast_forward
    global best_val_loss
    global logger
    global epoch

    rank = dist.get_rank()

    model = model.train()
    model = model.to(device)
    criterion = criterion.to(device)

    begin_of_epoch = time.perf_counter()
    start = time.perf_counter()

    if rank == 0:
        mean_loss = metrics.AverageMetric(name="loss")
        additional_losses = {}

        mean_forward_pass = metrics.AverageMetric(name="forward_pass")

        mean_bsz = metrics.AverageMetric(name="batch_size")
        batch_sizes = []
        batch_padded_seq_lengths = []

    train_loader_length = len(train_loader)
    logger.info(f"[rank:{rank}] [epoch:{epoch + 1}] train_loader_length: {train_loader_length}")

    for batch_num, batch in enumerate(train_loader):
        step += 1

        inputs, labels = prepare_batch(batch=batch,
                                       device=device,
                                       model_type=model_type)

        optimizer.zero_grad()

        start_forward = time.perf_counter()
        with amp.autocast(enabled=mixed_prec_scaler.is_enabled()):
            output, loss_dict = model(*inputs)
            end_forward = time.perf_counter()
            loss = criterion(output, labels)
            loss = loss + sum(loss_dict.values())

        mixed_prec_scaler.scale(loss).backward()
        mixed_prec_scaler.step(optimizer)
        mixed_prec_scaler.update()

        if rank == 0:
            mean_loss.add(loss.detach())
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in additional_losses:
                    additional_losses[loss_name] = metrics.AverageMetric(name=loss_name)
                additional_losses[loss_name].add(loss_value.detach())

            mean_forward_pass.add((end_forward - start_forward) * 1000)
            mean_bsz.add(labels.shape[0])
            batch_sizes.append(labels.shape[0])
            batch_padded_seq_lengths.append(inputs[0].shape[0])

        if lr_scheduler is not None:
            lr_scheduler.step()

        if step % eval_interval == 0 and rank == 0:
            val_loss = evaluate(model_type=model_type,
                                model=model,
                                encoder_tokenizer=encoder_tokenizer,
                                decoder_tokenizer=decoder_tokenizer,
                                val_loader=val_loader,
                                criterion=criterion,
                                text_metrics=text_metrics,
                                summary_writer=summary_writer,
                                device=device)
            io.save_checkpoint(checkpoint_path=os.path.join(checkpoint_dir, f"{model_name}-checkpoint-{step}.pt"),
                               model=model,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               grad_scaler=mixed_prec_scaler,
                               step=step,
                               epoch=epoch,
                               val_loss=val_loss)
            io.save_checkpoint(checkpoint_path=os.path.join(checkpoint_dir, f"{model_name}-checkpoint-last.pt"),
                               model=model,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               grad_scaler=mixed_prec_scaler,
                               step=step,
                               epoch=epoch,
                               val_loss=val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                io.save_checkpoint(checkpoint_path=os.path.join(checkpoint_dir, f"{model_name}-checkpoint-best.pt"),
                                   model=model,
                                   optimizer=optimizer,
                                   lr_scheduler=lr_scheduler,
                                   grad_scaler=mixed_prec_scaler,
                                   step=step,
                                   epoch=epoch,
                                   val_loss=val_loss)

            if keep_last_n_checkpoints > 0:
                # first get all checkpoints
                all_checkpoints = io.last_n_checkpoints(checkpoint_dir=checkpoint_dir, n=-1)
                for checkpoint in all_checkpoints[:-keep_last_n_checkpoints]:
                    os.remove(checkpoint)

            model = model.train()

        if step % log_interval == 0 and rank == 0:
            with StringIO() as buf, redirect_stdout(buf):
                GPUtil.showUtilization(all=True)
                util = buf.getvalue().strip()
                logger.info(f"[{step}] GPU utilization:\n{util}")

            if lr_scheduler is not None:
                lr = lr_scheduler.get_last_lr()[0]
                summary_writer.add_scalar("train_lr", lr, step)
                logger.info(f"[{step}] train_lr: {lr:.8f}")

            train_loss = mean_loss.calc()
            summary_writer.add_scalar("train_loss", train_loss, step)
            logger.info(f"[{step}] train_loss: {train_loss:.8f}")

            for loss_name, loss_metric in additional_losses.items():
                loss_value = loss_metric.calc()
                logger.info(f"[{step}] train_{loss_name}: {loss_value:.8f}")
                summary_writer.add_scalar(f"train_{loss_name}", loss_value, step)

            train_mean_bsz = mean_bsz.calc()
            summary_writer.add_scalar("train_mean_bsz", train_mean_bsz, step)
            logger.info(f"[{step}] train_mean_bsz: {train_mean_bsz:.8f}")

            train_mean_fwd = mean_forward_pass.calc()
            summary_writer.add_scalar("train_mean_fwd", train_mean_fwd, step)
            logger.info(f"[{step}] train_mean_fwd: {train_mean_fwd:.8f}")

            n_bins = 30
            summary_writer.add_histogram("train_bsz_hist",
                                         np.array(batch_sizes),
                                         step,
                                         bins=n_bins)

            summary_writer.add_histogram("train_batch_padded_seq_len_hist",
                                         np.array(batch_padded_seq_lengths),
                                         step,
                                         bins=n_bins)

            end = time.perf_counter()
            logger.info(f"[{step}] [train_time:{step - log_interval}\u2192{step}] {(end - start) / 60:.2f} minutes")
            logger.info(f"[{step}] [epoch:{epoch + 1}] "
                        f"{common.eta_minutes((end - begin_of_epoch) / 60, batch_num + 1, len(train_loader))}")

            mean_forward_pass.reset()
            mean_loss.reset()
            mean_bsz.reset()
            batch_sizes = []
            batch_padded_seq_lengths = []
            start = end


def evaluate(model_type: str,
             model: transformer.TransformerModel,
             encoder_tokenizer: tokenizers.Tokenizer,
             decoder_tokenizer: tokenizers.Tokenizer,
             val_loader: DataLoader,
             criterion: nn.Module,
             text_metrics: List[config.MetricConfig],
             summary_writer: SummaryWriter,
             device: torch.device) -> float:
    global step
    global logger

    model = model.eval()
    model = model.to(device)
    criterion = criterion.to(device)

    rank = dist.get_rank()
    assert rank == 0, "evaluation should be only done on main process"

    mean_loss = metrics.AverageMetric(name="loss")
    additional_losses = {}
    rand_batch_idx = torch.randint(low=0, high=len(val_loader), size=(1,)).item()

    tm: List[metrics.TextMetric] = [metrics.get_text_metric(metric_conf.name, **metric_conf.arguments)
                                    for metric_conf in text_metrics]
    start = time.perf_counter()

    with torch.no_grad():
        for batch_num, batch in enumerate(val_loader):

            inputs, labels = prepare_batch(batch=batch,
                                           device=device,
                                           model_type=model_type)

            output, loss_dict = model(*inputs)
            loss = criterion(output, labels)
            loss = loss + sum(loss_dict.values())

            mean_loss.add(loss.detach())
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in additional_losses:
                    additional_losses[loss_name] = metrics.AverageMetric(name=loss_name)
                additional_losses[loss_name].add(loss_value.detach())

            # evaluate text metrics always on the first batch (to see development across evaluations)
            # and one random batch (to see also other examples, see below)
            if batch_num == 0:
                for metric in tm:
                    metric.add(inputs=inputs,
                               outputs=output,
                               labels=labels,
                               encoder_tokenizer=encoder_tokenizer,
                               decoder_tokenizer=decoder_tokenizer)
                    text = metric.calc()
                    summary_writer.add_text(f"val_{metric.name()}",
                                            text,
                                            step)
                    logger.info(f"[{step}] val_{metric.name()}: {text}")

            if batch_num == rand_batch_idx:
                for metric in tm:
                    metric.add(inputs=inputs,
                               outputs=output,
                               labels=labels,
                               encoder_tokenizer=encoder_tokenizer,
                               decoder_tokenizer=decoder_tokenizer)
                    text = metric.calc()
                    summary_writer.add_text(f"val_{metric.name()}_rand",
                                            text,
                                            step)
                    logger.info(f"[{step}] val_{metric.name()}_rand: {text}")

    val_loss = mean_loss.calc()
    summary_writer.add_scalar("val_loss", val_loss, step)
    logger.info(f"[{step}] val_loss: {val_loss:.8f}")

    for loss_name, loss_metric in additional_losses.items():
        loss_value = loss_metric.calc()
        logger.info(f"[{step}] val_{loss_name}: {loss_value:.8f}")
        summary_writer.add_scalar(f"val_{loss_name}", loss_value, step)

    end = time.perf_counter()
    logger.info(f"[val_time:{step}] {(end - start) / 60:.2f} minutes")
    return val_loss


def train(rank: int,
          cfg: config.Config,
          directories: Dict[str, str],
          resuming: bool) -> None:
    global logger
    global epoch
    global step
    global steps_to_fast_forward
    global best_val_loss

    log_file = os.path.join(directories["experiment"], "logs.txt")
    common.add_file_log(logger, log_file)
    common.add_file_log(data.logger, log_file)

    device_props = torch.cuda.get_device_properties(rank)
    logger.info(f"[GPU:{rank}] {device_props.name}, {device_props.total_memory // 1024 // 1024:.0f}MiB "
                f"({device_props.major}.{device_props.minor}, {device_props.multi_processor_count})")

    if cfg.seed:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    model = transformer.get_model_from_config(config=cfg.model, device=device)
    encoder_tokenizer, decoder_tokenizer = transformer.get_tokenizers_from_model(model=model)

    pad_token_id = encoder_tokenizer.token_to_id(constants.PAD)
    train_loader, val_loader = data.get_data_from_config(train_config=cfg.train,
                                                         val_config=cfg.val,
                                                         seed=cfg.seed,
                                                         pad_token_id=pad_token_id)

    ignore_index = decoder_tokenizer.token_to_id(constants.PAD)
    criterion = loss.get_loss_from_config(config=cfg.train.loss,
                                          ignore_index=ignore_index,
                                          vocab_size=decoder_tokenizer.get_vocab_size())

    num_training_steps = len(train_loader) * cfg.train.num_epochs
    optimizer = get_optimizer_from_config(config=cfg.train.optimizer,
                                          model=model)

    lr_scheduler = lr_schedule.get_lr_scheduler_from_config(config=cfg.train.lr_scheduler,
                                                            num_training_steps=num_training_steps,
                                                            optimizer=optimizer) \
        if cfg.train.lr_scheduler is not None else None

    mixed_prec_scaler = amp.GradScaler(enabled=cfg.train.mixed_precision)

    summary_writer = SummaryWriter(log_dir=directories["tensorboard"])

    if rank == 0:
        logger.info(f"Using model:\n{model}")
        logger.info(f"Model parameters: {common.get_num_parameters(model)}")

        test_sentence = "This is a test sentence."
        if encoder_tokenizer is not None:
            logger.info(f"Testing encoder tokenizer: {encoder_tokenizer.encode(test_sentence, pair=None).tokens}")
        if decoder_tokenizer is not None:
            logger.info(f"Testing decoder tokenizer: {decoder_tokenizer.encode(test_sentence, pair=None).tokens}")

        logger.info(f"Type 'tensorboard --logdir {directories['tensorboard']}' "
                    f"to view the training process in Tensorboard")

    if resuming:
        last_checkpoint = io.last_n_checkpoints(directories["checkpoints"], 1)[0]
        checkpoint = io.load_checkpoint(last_checkpoint)
        io.load_state_dict(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if checkpoint.get("grad_scaler_state_dict") is not None:
            mixed_prec_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        steps_to_fast_forward = step % max(len(train_loader), 1)

        if rank == 0:
            logger.info(
                f"Resuming training from checkpoint {last_checkpoint}\n"
                f"Starting at epoch {epoch + 1} at step {step} with a best validation loss of {best_val_loss}\n"
                f"Need to fast forward {steps_to_fast_forward} steps"
            )

    ddp_model: transformer.TransformerModel = DDP(model, device_ids=[rank], output_device=rank)

    while epoch < cfg.train.num_epochs:
        train_loader.batch_sampler: data.DistributedDynamicSampler
        train_loader.batch_sampler.set_epoch(epoch)
        train_loader.batch_sampler.set_steps_to_fast_forward(steps_to_fast_forward)

        train_one_epoch(model=ddp_model,
                        encoder_tokenizer=encoder_tokenizer,
                        decoder_tokenizer=decoder_tokenizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        summary_writer=summary_writer,
                        device=device,
                        checkpoint_dir=directories["checkpoints"],
                        lr_scheduler=lr_scheduler,
                        mixed_prec_scaler=mixed_prec_scaler,
                        eval_interval=cfg.train.eval_interval,
                        log_interval=cfg.train.log_interval,
                        keep_last_n_checkpoints=cfg.train.keep_last_n_checkpoints,
                        text_metrics=cfg.val.text_metrics,
                        model_name=cfg.model.name,
                        model_type=cfg.model.type)

        epoch += 1
        steps_to_fast_forward = 0


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get("TORCH_DIST_PORT", "12345")

    # initialize the process group
    dist.init_process_group("nccl",
                            rank=rank,
                            world_size=world_size)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_distributed(rank: int,
                      world_size: int,
                      cfg: config.Config,
                      directories: Dict[str, str],
                      resuming: bool) -> None:
    setup_distributed(rank, world_size)

    train(rank, cfg, directories, resuming)

    cleanup_distributed()


def setup_experiment_dir(cfg: config.Config) -> str:
    experiment_name = re.sub(r"\s", "_", cfg.experiment)
    experiment_subdir = f"{experiment_name}-{datetime.today().strftime('%Y-%m-%d-%H_%M_%S')}"
    experiment_dir = os.path.join(cfg.experiment_dir, experiment_subdir)
    os.makedirs(experiment_dir)
    return experiment_dir


def check_gpus() -> int:
    world_size = torch.cuda.device_count()

    if not torch.cuda.is_available():
        logger.error("No available GPU detected. Exiting.")
        sys.exit(1)

    if world_size > 1:
        logger.info(f"Found {world_size} devices. Running distributed training.")

    return world_size


def main(args: argparse.Namespace) -> None:
    if args.config is not None:
        cfg = config.Config.from_yaml(args.config)
        experiment_dir = setup_experiment_dir(cfg)
        # save copy of config file to experiment directory
        with open(os.path.join(experiment_dir, "config.yaml"), "w", encoding="utf8") as f:
            f.write(str(cfg))
        resuming = False
    else:
        experiment_dir = args.resume
        cfg = config.Config.from_yaml(os.path.join(experiment_dir, "config.yaml"))
        if args.overwrite_train_data is not None:
            logger.info(f"Overriding train data to {args.overwrite_train_data}")
            cfg.train.train_data = args.overwrite_train_data
        if args.overwrite_val_data is not None:
            logger.info(f"Overriding val data to {args.overwrite_train_data}")
            cfg.val.val_data = args.overwrite_val_data
        resuming = True

    directories = {
        "experiment": experiment_dir,
        "checkpoints": os.path.join(experiment_dir, "checkpoints"),
        "tensorboard": os.path.join(experiment_dir, "tensorboard")
    }
    os.makedirs(directories["checkpoints"], exist_ok=True)
    os.makedirs(directories["tensorboard"], exist_ok=True)

    # check gpu environment
    world_size = check_gpus()

    # start distributed training
    mp.spawn(train_distributed,
             args=(world_size, cfg, directories, resuming),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
