"""Distributed AUDIO-LeJEPA pre-training entry point."""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets.audio_lejepa import make_audio_lejepa_loader
from models.video_lejepa import LeJEPALoss, ModelEMA, embedding_statistics
from models.audio_lejepa import load_emotion2vec_encoder
from models.audio_lejepa import build_audio_lejepa

LOGGER = logging.getLogger(__name__)


def _state():
    enabled = dist.is_available() and dist.is_initialized()
    return enabled, dist.get_rank() if enabled else 0, dist.get_world_size() if enabled else 1


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _optimizer(model, cfg):
    buckets = {name: [] for name in (
        "pretrained_decay", "pretrained_no_decay", "new_decay", "new_no_decay"
    )}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        pretrained = name.startswith("encoder.backbone.")
        no_decay = parameter.ndim <= 1 or "norm" in name.lower() or name.endswith("bias")
        buckets[f"{'pretrained' if pretrained else 'new'}_{'no_decay' if no_decay else 'decay'}"].append(parameter)
    groups = []
    for name, parameters in buckets.items():
        if not parameters:
            continue
        base_lr = float(cfg["lr_pretrained"] if name.startswith("pretrained") else cfg["lr_new"])
        groups.append({
            "params": parameters,
            "base_lr": base_lr,
            "lr": base_lr,
            "weight_decay": 0.0 if name.endswith("no_decay") else float(cfg["weight_decay"]),
            "group_name": name,
        })
    return torch.optim.AdamW(groups, betas=tuple(cfg.get("betas", (0.9, 0.98))), eps=float(cfg.get("eps", 1e-8)))


def _schedule(optimizer, step, cfg):
    warmup = int(cfg.get("warmup_steps", 0))
    total = int(cfg["total_steps"])
    if step < warmup:
        scale = (step + 1) / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        final = float(cfg.get("final_lr_scale", 0.0))
        scale = final + (1 - final) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * scale


def _reduce(value):
    value = value.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
        value.div_(dist.get_world_size())
    return float(value.cpu())


def _lrs(optimizer):
    values = {}
    for group in optimizer.param_groups:
        family = "lr_pretrained" if group["group_name"].startswith("pretrained") else "lr_new"
        values.setdefault(family, group["lr"])
    return values


def _append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _trim_history(path, maximum_step):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= maximum_step]
    if not rows:
        path.unlink()
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_curve(history_path, output_path):
    try:
        import matplotlib.pyplot as plt
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        figure, axis = plt.subplots(figsize=(8, 5))
        for key in ("loss", "invariance_loss", "sigreg_loss"):
            axis.plot([int(row["step"]) for row in rows], [float(row[key]) for row in rows], label=key)
        axis.set(xlabel="optimizer step", ylabel="loss", title="AUDIO-LeJEPA training")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    except Exception as error:
        LOGGER.warning("could not write AUDIO-LeJEPA curve: %s", error)


def save_checkpoint(path, model, optimizer, scaler, ema, step, epoch, args):
    raw = _unwrap(model)
    state = {
        "schema": "favor.audio_lejepa.v1",
        "model": raw.state_dict(),
        "encoder": raw.encoder.backbone.state_dict(),
        "projector": raw.projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "state_dict_ema": ema.state_dict()["shadow"] if ema is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "args": args,
    }
    temporary = Path(f"{path}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "favor.audio_lejepa.v1":
        raise ValueError(f"Not an AUDIO-LeJEPA checkpoint: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    return checkpoint


def main(args):
    distributed, rank, world_size = _state()
    is_main = rank == 0
    meta = args.get("meta", {})
    data_cfg = dict(args["data"])
    data_cfg["audio_augmentation"] = args.get("audio_augmentation", {})
    model_cfg = dict(args["model"])
    model_cfg["audio"] = {
        "sample_rate": int(data_cfg.get("sample_rate", 16000)),
        "process_seconds": float(data_cfg.get("global_seconds", 4.0)),
        "max_process_seconds": float(data_cfg.get("global_seconds", 4.0)),
    }
    opt_cfg = args["optimization"]
    loss_cfg = args.get("loss", {}).get("sigreg", {})
    ema_cfg = args.get("ema", {})
    seed = int(meta.get("seed", 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folder = Path(args["folder"])
    history_path = folder / "logs" / "history.csv"
    latest_path = folder / "latest.pt"
    if is_main:
        (folder / "logs").mkdir(parents=True, exist_ok=True)

    resume_path = meta.get("read_checkpoint")
    if not resume_path and meta.get("auto_resume", True) and latest_path.exists():
        resume_path = latest_path
    model = build_audio_lejepa(model_cfg)
    if not resume_path and (not distributed or is_main):
        init_cfg = model_cfg.get("init", {})
        if init_cfg.get("mode", "emotion2vec") != "emotion2vec":
            raise ValueError("AUDIO-LeJEPA model.init.mode must be emotion2vec")
        load_emotion2vec_encoder(
            model.encoder,
            init_cfg.get("checkpoint", "CKPT/emotion2vec_plus_large/model.pt"),
            init_cfg.get("min_load_ratio", 0.95),
        )
    model.to(device)
    optimizer = _optimizer(model, opt_cfg)
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    mixed = device.type == "cuda" and dtype_name in {"bfloat16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype_name == "float16")
    ema = ModelEMA(
        model,
        decay=float(ema_cfg.get("decay", 0.9999)),
        update_every=int(ema_cfg.get("update_every", 32)),
        device=ema_cfg.get("device", device),
    ) if ema_cfg.get("enabled", True) else None
    criterion = LeJEPALoss(
        sigreg_weight=loss_cfg.get("weight", 0.02),
        knots=loss_cfg.get("knots", 17),
        num_proj=loss_cfg.get("num_proj", 1024),
        normalize_by_n=loss_cfg.get("normalize_by_n", False),
    ).to(device)

    step = epoch = 0
    if resume_path:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scaler, ema)
        step, epoch = int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))
        if is_main:
            _trim_history(history_path, step)
        LOGGER.info("resumed AUDIO-LeJEPA from %s at step=%d", resume_path, step)
    elif is_main:
        save_checkpoint(folder / "init.pt", model, optimizer, scaler, ema, 0, 0, args)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[0] if device.type == "cuda" else None)
    _, loader, sampler = make_audio_lejepa_loader(data_cfg, rank=rank, world_size=world_size)
    if not len(loader):
        raise ValueError("AUDIO-LeJEPA DataLoader is empty; reduce data.batch_size")
    total_steps = int(opt_cfg["total_steps"])
    accumulation = max(1, int(opt_cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(meta.get("log_every_steps", 50)))
    save_every = int(meta.get("save_every_steps", 5000))
    if total_steps < 1:
        raise ValueError("optimization.total_steps must be at least 1")
    if save_every < 0:
        raise ValueError("meta.save_every_steps must be non-negative")
    criterion.train()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.perf_counter()
    samples_since_log = 0

    while step < total_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            global_audio = batch["global_audio"].to(device, non_blocking=True)
            global_lengths = batch["global_lengths"].to(device, non_blocking=True)
            local_audio = batch["local_audio"].to(device, non_blocking=True)
            local_lengths = batch["local_lengths"].to(device, non_blocking=True)
            samples_since_log += int(global_audio.shape[0])
            should_step = (micro_step + 1) % accumulation == 0
            sync = nullcontext() if should_step or not hasattr(model, "no_sync") else model.no_sync()
            with sync:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed):
                    embeddings = model(global_audio, global_lengths, local_audio, local_lengths)
                    losses = criterion(embeddings)
                    scaled_loss = losses["loss"] / accumulation
                scaler.scale(scaled_loss).backward()
            micro_step += 1
            if not should_step:
                continue

            _schedule(optimizer, step, opt_cfg)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("clip_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                ema.update(model, step)

            if step == 1 or step % log_every == 0 or step == total_steps:
                with torch.no_grad():
                    stats = embedding_statistics(embeddings.detach())
                    global_embedding, local_embedding = embeddings[:, 0], embeddings[:, 1:]
                    cosine = torch.nn.functional.cosine_similarity(
                        global_embedding[:, None, :], local_embedding, dim=-1
                    ).mean()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    occurrence = batch["augmentation_occurrence"].to(device).float().mean((0, 1))
                    local_duration = local_lengths.float().mean() / int(data_cfg.get("sample_rate", 16000))
                    global_duration = global_lengths.float().mean() / int(data_cfg.get("sample_rate", 16000))
                    row = {
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": _reduce(losses["loss"]),
                        "invariance_loss": _reduce(losses["invariance_loss"]),
                        "sigreg_loss": _reduce(losses["sigreg_loss"]),
                        "global_embedding_std": _reduce(global_embedding.float().std(dim=0, unbiased=False).mean()),
                        "local_embedding_std": _reduce(local_embedding.float().reshape(-1, local_embedding.shape[-1]).std(dim=0, unbiased=False).mean()),
                        "cosine_similarity": _reduce(cosine),
                        "embedding_std": _reduce(stats["embedding_std"]),
                        "rankme": _reduce(stats["rankme"]),
                        "global_duration": _reduce(global_duration),
                        "local_duration": _reduce(local_duration),
                        "gain_rate": _reduce(occurrence[0]),
                        "speed_rate": _reduce(occurrence[1]),
                        "polarity_rate": _reduce(occurrence[2]),
                        "time_mask_rate": _reduce(occurrence[3]),
                        "repeat_ratio": _reduce(batch["was_repeated"].float().mean().to(device)),
                        "pad_ratio": _reduce(batch["was_padded"].float().mean().to(device)),
                        **_lrs(optimizer),
                        "ema_decay": ema.decay if ema else 0.0,
                        "grad_norm": _reduce(grad_norm),
                        "samples_per_second": samples_since_log * world_size / elapsed,
                    }
                if is_main:
                    _append_csv(history_path, row)
                    LOGGER.info(
                        "step=%d/%d loss=%.5f inv=%.5f sigreg=%.5f gstd=%.4f lstd=%.4f "
                        "cos=%.4f rank=%.2f grad=%.3f samples/s=%.2f",
                        step, total_steps, row["loss"], row["invariance_loss"], row["sigreg_loss"],
                        row["global_embedding_std"], row["local_embedding_std"], row["cosine_similarity"],
                        row["rankme"], row["grad_norm"], row["samples_per_second"],
                    )
                if not math.isfinite(row["loss"]):
                    raise FloatingPointError(f"non-finite AUDIO-LeJEPA loss: {row['loss']}")
                started = time.perf_counter()
                samples_since_log = 0
            if is_main and save_every > 0 and step % save_every == 0:
                save_checkpoint(latest_path, model, optimizer, scaler, ema, step, epoch, args)
                save_checkpoint(folder / f"step-{step}.pt", model, optimizer, scaler, ema, step, epoch, args)
                _write_curve(history_path, folder / "logs" / "metrics_curves.png")
            if step >= total_steps:
                break
        epoch += 1

    if is_main:
        save_checkpoint(latest_path, model, optimizer, scaler, ema, step, epoch, args)
        _write_curve(history_path, folder / "logs" / "metrics_curves.png")
    if distributed:
        dist.barrier()
    return {"latest_checkpoint": latest_path}

