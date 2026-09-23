"""Distributed audio JEPA pre-training entry point."""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets.audio_jepa.pretrain_dataset import make_audio_pretrain_loader
from models.pretrain_a_model import build_audio_jepa
from utils.audio_checkpoint import (
    init_from_emotion2vec,
    restore_audio_pretrain_checkpoint,
    save_audio_pretrain_checkpoint,
)
from utils.audio_pretrain_logging import (
    load_audio_pretrain_history,
    save_audio_pretrain_curves,
    write_audio_pretrain_history,
)

LOGGER = logging.getLogger(__name__)


def _state():
    enabled = dist.is_available() and dist.is_initialized()
    return enabled, dist.get_rank() if enabled else 0, dist.get_world_size() if enabled else 1


def _optimizer(model, cfg):
    groups = {
        "pretrained_decay": [], "pretrained_no_decay": [],
        "new_decay": [], "new_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        pretrained = name.startswith("encoder.")
        no_decay = parameter.ndim <= 1 or "norm" in name.lower() or name.endswith("bias")
        family = "pretrained" if pretrained else "new"
        decay = "no_decay" if no_decay else "decay"
        groups[f"{family}_{decay}"].append(parameter)
    weight_decay = float(cfg["weight_decay"])
    pretrained_lr = float(cfg["lr_pretrained"])
    new_lr = float(cfg["lr_new"])
    parameter_groups = []
    for name, parameters in groups.items():
        if not parameters:
            continue
        base_lr = pretrained_lr if name.startswith("pretrained") else new_lr
        parameter_groups.append({
            "params": parameters,
            "base_lr": base_lr,
            "lr": base_lr,
            "weight_decay": 0.0 if name.endswith("no_decay") else weight_decay,
            "group_name": name,
        })
    return torch.optim.AdamW(
        parameter_groups,
        betas=tuple(cfg.get("betas", (0.9, 0.98))),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _lr_step(optimizer, step, cfg):
    warmup = int(cfg.get("warmup_steps", 0))
    total = int(cfg["total_steps"])
    final_scale = float(cfg.get("final_lr_scale", 0.0))
    if step < warmup:
        scale = step / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total - warmup)
        scale = final_scale + (1 - final_scale) * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * scale


@torch.no_grad()
def _ema_update(target, source, decay):
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.data.mul_(decay).add_(source_parameter.data, alpha=1.0 - decay)


def _install_file_logger(path, append):
    path = Path(path).resolve()
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == path:
            return
    handler = logging.FileHandler(path, mode="a" if append else "w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)


def _learning_rates(optimizer):
    pretrained = [
        group["lr"] for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("pretrained")
    ]
    new = [
        group["lr"] for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("new")
    ]
    return (
        float(pretrained[0] if pretrained else optimizer.param_groups[0]["lr"]),
        float(new[0] if new else optimizer.param_groups[-1]["lr"]),
    )


def _empty_window():
    return {
        "loss": 0.0,
        "context_ratio": 0.0,
        "target_ratio": 0.0,
        "pad_ratio": 0.0,
        "target_std": 0.0,
        "micro_batches": 0.0,
        "samples": 0.0,
        "grad_norm": 0.0,
        "optimizer_updates": 0.0,
    }


def _reduce_window(window, elapsed, device, distributed, world_size):
    fields = tuple(window)
    values = torch.tensor([window[field] for field in fields], dtype=torch.float64, device=device)
    elapsed_value = torch.tensor(float(elapsed), dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_value, op=dist.ReduceOp.MAX)
    reduced = dict(zip(fields, values.cpu().tolist()))
    micro_batches = max(1.0, reduced["micro_batches"])
    optimizer_updates = max(1.0, reduced["optimizer_updates"] / max(1, world_size))
    elapsed_seconds = max(1e-12, float(elapsed_value.cpu()))
    return {
        "loss": reduced["loss"] / micro_batches,
        "context_ratio": reduced["context_ratio"] / micro_batches,
        "target_ratio": reduced["target_ratio"] / micro_batches,
        "pad_ratio": reduced["pad_ratio"] / micro_batches,
        "target_std": reduced["target_std"] / micro_batches,
        "grad_norm": reduced["grad_norm"] / max(1.0, reduced["optimizer_updates"]),
        "step_time": elapsed_seconds / optimizer_updates,
        "samples_per_second": reduced["samples"] / elapsed_seconds,
    }


def main(args):
    distributed, rank, world_size = _state()
    is_main = rank == 0
    meta = args.get("meta", {})
    data_cfg = args["data"]
    model_cfg = dict(args["model"])
    model_cfg["audio"] = {
        key: data_cfg[key] for key in (
            "sample_rate", "process_seconds", "min_process_seconds", "max_process_seconds"
        ) if key in data_cfg
    }
    model_cfg["mask"] = args["mask"]
    opt_cfg = args["optimization"]
    seed = int(meta.get("seed", 0))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folder = Path(args["folder"])
    logs_dir = folder / "logs"
    history_path = logs_dir / "history.csv"
    curves_path = logs_dir / "metrics_curves.png"
    latest_path = folder / "latest.pt"
    explicit_resume = meta.get("read_checkpoint")
    resume_path = Path(explicit_resume) if explicit_resume else None
    if resume_path is None and meta.get("auto_resume", True) and latest_path.is_file():
        resume_path = latest_path
    if is_main:
        folder.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        _install_file_logger(folder / "train.log", append=resume_path is not None)

    model = build_audio_jepa(model_cfg)

    step = epoch = 0
    checkpoint = None
    if resume_path is not None:
        checkpoint = restore_audio_pretrain_checkpoint(resume_path, model, strict=True)
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        LOGGER.info("Resumed audio pre-training from %s at step=%d", resume_path, step)
    else:
        init_path = meta.get("init_checkpoint")
        if not init_path:
            raise ValueError("A new pretrain_a run requires meta.init_checkpoint")
        # DDP synchronizes rank 0 parameters in its constructor. Loading this
        # 1.9 GB initialization checkpoint once avoids simultaneous disk reads.
        if not distributed or is_main:
            init_from_emotion2vec(model, init_path, model_cfg)

    history = []
    if is_main:
        if resume_path is not None:
            history = load_audio_pretrain_history(history_path, max_step=step)
            if not history and checkpoint is not None:
                history = [row for row in checkpoint.get("history", []) if int(row["step"]) <= step]
        write_audio_pretrain_history(history_path, history)

    model.to(device)
    optimizer = _optimizer(model, opt_cfg)
    if resume_path is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif is_main:
        save_audio_pretrain_checkpoint(
            folder / "init.pt", model=model, optimizer=optimizer, step=0, epoch=0,
            args=args, history=history,
        )
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[0] if device.type == "cuda" else None, find_unused_parameters=False
        )
    raw_model = model.module if distributed else model
    _, loader, sampler = make_audio_pretrain_loader(data_cfg, rank=rank, world_size=world_size)
    if not len(loader):
        raise ValueError("Audio pre-training DataLoader is empty; reduce data.batch_size")
    total_steps = int(opt_cfg["total_steps"])
    accumulation = int(opt_cfg.get("grad_accum_steps", 1))
    log_every = int(meta.get("log_every_steps", 50))
    save_every = int(meta.get("save_every_steps", 5000))
    if total_steps < 1:
        raise ValueError("optimization.total_steps must be at least 1")
    if accumulation < 1:
        raise ValueError("optimization.grad_accum_steps must be at least 1")
    if log_every < 1:
        raise ValueError("meta.log_every_steps must be at least 1")
    if save_every < 0:
        raise ValueError("meta.save_every_steps must be non-negative")
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    mixed = device.type == "cuda" and dtype_name in {"bfloat16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype_name == "float16")
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window = _empty_window()
    window_started = time.perf_counter()

    while step < total_steps:
        sampler.set_epoch(epoch)
        for waveforms, lengths in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            _lr_step(optimizer, step, opt_cfg)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed):
                loss, logs = model(waveforms, valid_wave_lens=lengths, step=step, total_steps=total_steps)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            window["loss"] += float(loss.detach())
            window["context_ratio"] += float(logs["context_ratio"])
            window["target_ratio"] += float(logs["target_ratio"])
            window["pad_ratio"] += float(logs["pad_ratio"])
            window["target_std"] += float(logs["target_std"])
            window["micro_batches"] += 1
            window["samples"] += int(waveforms.shape[0])
            micro_step += 1
            if micro_step % accumulation:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(opt_cfg.get("clip_grad_norm", 1.0))
            )
            window["grad_norm"] += float(grad_norm.detach())
            window["optimizer_updates"] += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            ema_cfg = opt_cfg["ema"]
            progress = min(1.0, step / max(1, int(ema_cfg["anneal_steps"])))
            decay = float(ema_cfg["start"]) + progress * (float(ema_cfg["end"]) - float(ema_cfg["start"]))
            _ema_update(raw_model.target_encoder.context_encoder, raw_model.encoder.context_encoder, decay)
            step += 1
            log_due = step == 1 or step % log_every == 0 or step >= total_steps
            if log_due:
                summary = _reduce_window(
                    window, time.perf_counter() - window_started,
                    device, distributed, world_size,
                )
                if is_main:
                    lr_pretrained, lr_new = _learning_rates(optimizer)
                    row = {
                        "step": step,
                        "epoch": epoch + 1,
                        **summary,
                        "lr_pretrained": lr_pretrained,
                        "lr_new": lr_new,
                        "ema_decay": decay,
                    }
                    history = [item for item in history if int(item["step"]) < step]
                    history.append(row)
                    write_audio_pretrain_history(history_path, history)
                    LOGGER.info(
                        "step=%d/%d epoch=%d loss=%.5f context=%.3f target=%.3f "
                        "padding=%.3f target_std=%.4f lr_pretrained=%.3e lr_new=%.3e "
                        "ema=%.6f grad_norm=%.4f step_time=%.3fs samples/s=%.2f",
                        step, total_steps, epoch + 1, row["loss"], row["context_ratio"],
                        row["target_ratio"], row["pad_ratio"], row["target_std"],
                        row["lr_pretrained"], row["lr_new"], row["ema_decay"],
                        row["grad_norm"], row["step_time"], row["samples_per_second"],
                    )
                window = _empty_window()
                window_started = time.perf_counter()
            if is_main and save_every > 0 and step % save_every == 0:
                save_audio_pretrain_checkpoint(
                    latest_path, model=model, optimizer=optimizer,
                    step=step, epoch=epoch, args=args, history=history,
                )
                save_audio_pretrain_curves(history, curves_path)
            if step >= total_steps:
                break
        epoch += 1

    if is_main:
        save_audio_pretrain_checkpoint(
            latest_path, model=model, optimizer=optimizer, step=step, epoch=epoch,
            args=args, history=history,
        )
        save_audio_pretrain_curves(history, curves_path)
    if distributed:
        dist.barrier()
    return {"latest_checkpoint": latest_path}
