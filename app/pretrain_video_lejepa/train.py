from __future__ import annotations

import csv
import logging
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets.video_lejepa import make_video_lejepa_dataset
from models.video_lejepa import (
    LeJEPALoss,
    ModelEMA,
    build_video_lejepa,
    embedding_statistics,
    load_vjepa_encoder,
)
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _reduce_mean(value):
    value = value.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(dist.get_world_size())
    return value


def _cosine(start, end, progress):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def _schedule(step, total_steps, warmup_steps, start_lr, lr, final_lr, wd, final_wd):
    if warmup_steps and step <= warmup_steps:
        current_lr = start_lr + (lr - start_lr) * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        current_lr = _cosine(lr, final_lr, progress)
    current_wd = _cosine(wd, final_wd, step / max(total_steps, 1))
    return current_lr, current_wd


def _set_optimizer_values(optimizer, lr, wd):
    for group in optimizer.param_groups:
        group["lr"] = lr
        if not group.get("WD_exclude", False):
            group["weight_decay"] = wd


def _make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True},
        ],
        lr=float(cfg.get("lr", 4e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.04)),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _trim_history(path, checkpoint_epoch, checkpoint_step):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [
            row for row in reader
            if int(row["epoch"]) <= checkpoint_epoch
            and int(row["global_step"]) <= checkpoint_step
        ]
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_training_curve(history_path, output_path):
    """Best-effort loss curve; CSV remains the authoritative log."""
    try:
        import matplotlib.pyplot as plt

        steps, total, invariance, sigreg = [], [], [], []
        with Path(history_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                steps.append(int(row["global_step"]))
                total.append(float(row["loss"]))
                invariance.append(float(row["invariance_loss"]))
                sigreg.append(float(row["sigreg_loss"]))
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(steps, total, label="total")
        axis.plot(steps, invariance, label="invariance")
        axis.plot(steps, sigreg, label="SIGReg")
        axis.set(xlabel="optimizer step", ylabel="loss", title="VIDEO-LeJEPA training")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    except Exception as error:
        logger.warning("could not write training curve: %s", error)


def save_checkpoint(path, model, optimizer, scaler, ema, epoch, global_step, args):
    ema_state = None if ema is None else ema.state_dict()
    state = {
        "schema": "favor.video_lejepa.v1",
        "model": _unwrap(model).state_dict(),
        "encoder": _unwrap(model).encoder.state_dict(),
        "projector": _unwrap(model).projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "ema": ema_state,
        # Direct shadow mapping for downstream evaluators, matching the
        # LeVJEPA reference checkpoint convention.
        "state_dict_ema": None if ema_state is None else ema_state["shadow"],
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": args,
    }
    temporary = f"{path}.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(path, model, optimizer=None, scaler=None, ema=None, weights_only=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    message = _unwrap(model).load_state_dict(checkpoint["model"], strict=True)
    if not weights_only:
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
    return (0, 0, message) if weights_only else (
        int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0)), message
    )


def main(args):
    meta = args.get("meta", {})
    data_cfg = args.get("data", {})
    model_cfg = dict(args.get("model", {}))
    opt_cfg = args.get("optimization", {})
    loss_cfg = args.get("loss", {}).get("sigreg", {})
    ema_cfg = args.get("ema", {})
    folder = Path(args["folder"])
    folder.mkdir(parents=True, exist_ok=True)

    world_size, rank = init_distributed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = int(meta.get("seed", 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_frames = int(data_cfg.get("num_frames", (data_cfg.get("dataset_fpcs") or [16])[0]))
    model_cfg.setdefault("num_frames", num_frames)
    model_cfg.setdefault("global_size", data_cfg.get("global_size", data_cfg.get("crop_size", 224)))
    if int(model_cfg.get("tubelet_size", 1)) != 1:
        raise ValueError("VIDEO-LeJEPA benchmark requires model.tubelet_size=1")
    model = build_video_lejepa(model_cfg)
    init_cfg = model_cfg.get("init", {})
    init_mode = init_cfg.get("mode", "vjepa_checkpoint")
    if init_mode == "vjepa_checkpoint":
        checkpoint_path = init_cfg.get("checkpoint", "CKPT/vjepa2/vitl.pt")
        load_vjepa_encoder(model.encoder, checkpoint_path, init_cfg.get("min_load_ratio", 0.90))
    elif init_mode != "scratch":
        raise ValueError(f"model.init.mode must be vjepa_checkpoint or scratch, got {init_mode!r}")
    model.to(device)

    optimizer = _make_optimizer(model, opt_cfg)
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    amp_enabled = device.type == "cuda" and dtype_name in ("bfloat16", "float16")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    ema = ModelEMA(
        model,
        decay=ema_cfg.get("decay", 0.9999),
        update_every=ema_cfg.get("update_every", 32),
        device=ema_cfg.get("device", device),
    ) if ema_cfg.get("enabled", True) else None
    criterion = LeJEPALoss(
        sigreg_weight=loss_cfg.get("weight", 0.02),
        knots=loss_cfg.get("knots", 17),
        num_proj=loss_cfg.get("num_proj", 1024),
        normalize_by_n=loss_cfg.get("normalize_by_n", False),
    ).to(device)

    _, loader, sampler = make_video_lejepa_dataset(data_cfg, rank=rank, world_size=world_size)
    accumulation = max(1, int(opt_cfg.get("accumulate_grad_batches", 1)))
    ipe = int(opt_cfg.get("ipe") or len(loader))
    epochs = int(opt_cfg.get("epochs", 1))
    optimizer_steps_per_epoch = math.ceil(ipe / accumulation)
    total_steps = max(1, int(opt_cfg.get("ipe_scale", 1.0) * epochs * optimizer_steps_per_epoch))
    warmup_steps = int(float(opt_cfg.get("warmup", 0)) * optimizer_steps_per_epoch)

    start_epoch = global_step = 0
    history_path = folder / "history.csv"
    latest = folder / "latest.pt"
    resume_path = meta.get("read_checkpoint")
    if meta.get("load_checkpoint", True) and latest.exists():
        resume_path = latest
    if resume_path and Path(resume_path).exists() and Path(resume_path) != Path(init_cfg.get("checkpoint", "")):
        reset_epoch = bool(meta.get("reset_epoch", False))
        start_epoch, global_step, message = load_training_checkpoint(
            resume_path, model, optimizer, scaler, ema, weights_only=reset_epoch
        )
        if rank == 0 and not reset_epoch:
            _trim_history(history_path, start_epoch, global_step)
        logger.info("resumed VIDEO-LeJEPA from %s: %s", resume_path, message)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[0] if device.type == "cuda" else None)

    save_every = int(meta.get("save_every_freq", 1))
    log_freq = int(meta.get("log_freq", 10))
    clip_grad = float(opt_cfg.get("clip_grad_norm", 1.0))
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        window_start = time.perf_counter()
        for iteration in range(ipe):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            global_video = batch["global_video"].to(device, non_blocking=True)
            local_video = batch["local_video"].to(device, non_blocking=True)
            should_step = (iteration + 1) % accumulation == 0 or iteration + 1 == ipe
            sync_context = nullcontext() if should_step or not hasattr(model, "no_sync") else model.no_sync()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    embeddings = model(global_video, local_video)
                    losses = criterion(embeddings)
                    scaled_loss = losses["loss"] / accumulation
                scaler.scale(scaled_loss).backward()

            grad_norm = torch.tensor(float("nan"), device=device)
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                global_step += 1
                current_lr, current_wd = _schedule(
                    global_step, total_steps, warmup_steps,
                    float(opt_cfg.get("start_lr", 1e-4)), float(opt_cfg.get("lr", 4e-4)),
                    float(opt_cfg.get("final_lr", opt_cfg.get("lr", 4e-4))),
                    float(opt_cfg.get("weight_decay", 0.04)),
                    float(opt_cfg.get("final_weight_decay", opt_cfg.get("weight_decay", 0.04))),
                )
                _set_optimizer_values(optimizer, current_lr, current_wd)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model, global_step)

            stats = embedding_statistics(embeddings.detach())
            elapsed = max(time.perf_counter() - window_start, 1e-9)
            clips_per_second = global_video.shape[0] * world_size / elapsed
            window_start = time.perf_counter()
            row = {
                "epoch": epoch + 1,
                "iteration": iteration,
                "global_step": global_step,
                "loss": float(_reduce_mean(losses["loss"])),
                "invariance_loss": float(_reduce_mean(losses["invariance_loss"])),
                "sigreg_loss": float(_reduce_mean(losses["sigreg_loss"])),
                "embedding_std": float(_reduce_mean(stats["embedding_std"])),
                "rankme": float(_reduce_mean(stats["rankme"])),
                "lr": current_lr,
                "weight_decay": current_wd,
                "ema_decay": ema.decay if ema else 0.0,
                "grad_norm": float(_reduce_mean(grad_norm)),
                "clips_per_second": clips_per_second,
            }
            if rank == 0:
                _append_csv(history_path, row)
                if iteration % log_freq == 0 or iteration + 1 == ipe:
                    logger.info(
                        "[ep %d it %d] loss=%.5f inv=%.5f sigreg=%.5f std=%.4f rank=%.2f lr=%.2e wd=%.3g grad=%.3f clips/s=%.2f",
                        epoch + 1, iteration, row["loss"], row["invariance_loss"], row["sigreg_loss"],
                        row["embedding_std"], row["rankme"], current_lr, current_wd, row["grad_norm"], clips_per_second,
                    )
            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"non-finite VIDEO-LeJEPA loss: {row['loss']}")

        if rank == 0:
            save_checkpoint(latest, model, optimizer, scaler, ema, epoch + 1, global_step, args)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(folder / f"e{epoch + 1}.pt", model, optimizer, scaler, ema, epoch + 1, global_step, args)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if rank == 0:
        _write_training_curve(history_path, folder / "training_curve.png")
    return _unwrap(model)
