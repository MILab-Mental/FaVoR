"""Distributed audio JEPA pre-training entry point."""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets.audio_pretrain_dataset import make_audio_pretrain_loader
from models.pretrain_a_model import build_audio_jepa
from utils.audio_checkpoint import (
    init_from_emotion2vec,
    restore_audio_pretrain_checkpoint,
    save_audio_pretrain_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def _state():
    enabled = dist.is_available() and dist.is_initialized()
    return enabled, dist.get_rank() if enabled else 0, dist.get_world_size() if enabled else 1


def _optimizer(model, cfg):
    groups = {"pretrained": [], "new": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "norm" in name.lower() or name.endswith("bias"):
            groups["no_decay"].append(parameter)
        elif name.startswith("encoder.feature_extractor") or name.startswith("encoder.context_encoder"):
            groups["pretrained"].append(parameter)
        else:
            groups["new"].append(parameter)
    return torch.optim.AdamW([
        {"params": groups["pretrained"], "base_lr": float(cfg["lr_pretrained"]),
         "lr": float(cfg["lr_pretrained"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": groups["new"], "base_lr": float(cfg["lr_new"]),
         "lr": float(cfg["lr_new"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": groups["no_decay"], "base_lr": float(cfg["lr_new"]),
         "lr": float(cfg["lr_new"]), "weight_decay": 0.0},
    ], betas=tuple(cfg.get("betas", (0.9, 0.98))), eps=float(cfg.get("eps", 1e-8)))


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
    if is_main:
        folder.mkdir(parents=True, exist_ok=True)

    model = build_audio_jepa(model_cfg)
    latest_path = folder / "latest.pt"
    explicit_resume = meta.get("read_checkpoint")
    resume_path = Path(explicit_resume) if explicit_resume else None
    if resume_path is None and meta.get("auto_resume", True) and latest_path.is_file():
        resume_path = latest_path

    step = epoch = 0
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

    model.to(device)
    optimizer = _optimizer(model, opt_cfg)
    if resume_path is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif is_main:
        save_audio_pretrain_checkpoint(
            folder / "init.pt", model=model, optimizer=optimizer, step=0, epoch=0, args=args
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
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    mixed = device.type == "cuda" and dtype_name in {"bfloat16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype_name == "float16")
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    micro_step = 0

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
            running_loss += float(loss.detach())
            micro_step += 1
            if micro_step % accumulation:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("clip_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            ema_cfg = opt_cfg["ema"]
            progress = min(1.0, step / max(1, int(ema_cfg["anneal_steps"])))
            decay = float(ema_cfg["start"]) + progress * (float(ema_cfg["end"]) - float(ema_cfg["start"]))
            _ema_update(raw_model.target_encoder.context_encoder, raw_model.encoder.context_encoder, decay)
            step += 1
            if is_main and (step == 1 or step % log_every == 0):
                LOGGER.info(
                    "step=%d/%d loss=%.5f context=%.3f target=%.3f padding=%.3f lr=%.3e ema=%.6f",
                    step, total_steps, running_loss / (1 if step == 1 else log_every),
                    float(logs["context_ratio"]), float(logs["target_ratio"]),
                    float(logs["pad_ratio"]), optimizer.param_groups[0]["lr"], decay,
                )
                running_loss = 0.0
            if is_main and save_every > 0 and step % save_every == 0:
                save_audio_pretrain_checkpoint(
                    folder / f"s{step}.pt", model=model, optimizer=optimizer, step=step, epoch=epoch, args=args
                )
                save_audio_pretrain_checkpoint(
                    latest_path, model=model, optimizer=optimizer, step=step, epoch=epoch, args=args
                )
            if step >= total_steps:
                break
        epoch += 1

    if is_main:
        save_audio_pretrain_checkpoint(
            latest_path, model=model, optimizer=optimizer, step=step, epoch=epoch, args=args
        )
    if distributed:
        dist.barrier()
    return {"latest_checkpoint": latest_path}
