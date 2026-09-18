"""Generic distributed audio fine-tuning for classification and regression."""

from __future__ import annotations

import csv
import logging
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.audio_finetune_dataset import make_audiodataset_finetune_a
from models.finetune_a_model import AudioFineTuneModel
from utils.audio_checkpoint import load_audio_backbone
from utils.classification_metrics import classification_metrics, multilabel_metrics
from utils.progress import progress_ncols

LOGGER = logging.getLogger(__name__)


class DistributedEvaluationSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.dataset, self.rank, self.world_size = dataset, rank, world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def _state():
    enabled = dist.is_available() and dist.is_initialized()
    return enabled, dist.get_rank() if enabled else 0, dist.get_world_size() if enabled else 1


def _raw(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _gather(value, distributed, world_size):
    if not distributed:
        return [value]
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _optimizer(model, cfg, frozen):
    if frozen:
        parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(parameters, lr=float(cfg["head_lr"]),
                                 weight_decay=float(cfg.get("weight_decay", 0.0)),
                                 betas=tuple(cfg.get("betas", (0.9, 0.999))))
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["backbone_lr"])},
        {"params": model.head.parameters(), "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg.get("weight_decay", 0.0)),
       betas=tuple(cfg.get("betas", (0.9, 0.999))))


def _checkpoint(path, model, optimizer, scheduler, epoch, best_score, args):
    torch.save({
        "schema_version": 1, "modality": "audio", "stage": "finetune",
        "epoch": epoch, "encoder": _raw(model).backbone.state_dict(),
        "task_head": _raw(model).head.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "best_score": best_score, "args": args,
    }, path)


def _restore(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(checkpoint["encoder"], strict=True)
    model.head.load_state_dict(checkpoint["task_head"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _run_epoch(model, loader, criterion, optimizer, scaler, device, task, training, mixed, amp_dtype,
               *, epoch, epochs, is_main):
    model.train(training)
    if not any(parameter.requires_grad for parameter in _raw(model).backbone.parameters()):
        _raw(model).backbone.eval()
    total_loss, count = 0.0, 0
    paths, truths, outputs = [], [], []
    context = torch.enable_grad if training else torch.no_grad
    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        disable=not is_main,
        desc=f"{phase} {epoch + 1}/{epochs}",
        ncols=progress_ncols(),
    )
    with context():
        for clips, labels, batch_paths in progress:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if task == "classification":
                labels = labels.long()
            else:
                labels = labels.float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed):
                logits = model(clips)
                loss = criterion(logits if task != "regression" else logits.squeeze(-1), labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
            if is_main:
                progress.set_postfix(loss=f"{total_loss / max(1, count):.4f}")
            paths.extend(batch_paths)
            truths.extend(labels.detach().cpu().tolist())
            outputs.append(logits.detach().float().cpu().numpy())
    return total_loss, count, paths, truths, outputs


def main(args):
    distributed, rank, world_size = _state()
    is_main = rank == 0
    meta, data_cfg, opt_cfg = args.get("meta", {}), args["data"], args["optimization"]
    model_cfg = dict(args["model"])
    model_cfg["audio"] = {
        "sample_rate": data_cfg.get("sample_rate", 16000),
        "process_seconds": data_cfg.get("process_seconds", 4.0),
        "max_process_seconds": data_cfg.get("process_seconds", 4.0),
    }
    seed = int(meta.get("seed", 0))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folder = Path(args["folder"])
    if is_main:
        folder.mkdir(parents=True, exist_ok=True)

    task = data_cfg.get("task", "classification").lower()
    num_class = data_cfg.get("num_class")
    train_dataset, val_dataset = make_audiodataset_finetune_a(
        data_cfg["datasets"], root_paths=data_cfg["rootpaths"],
        label_column=data_cfg["label_column"], task=task, num_class=num_class,
        sample_rate=data_cfg.get("sample_rate", 16000),
        process_seconds=data_cfg.get("process_seconds", 4.0),
        num_clips=data_cfg.get("num_clips", 1),
    )
    train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True) if distributed else None
    val_sampler = DistributedEvaluationSampler(val_dataset, rank, world_size) if distributed else None
    loader_args = dict(
        batch_size=data_cfg["batch_size"], num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_mem", False),
        persistent_workers=bool(data_cfg.get("persistent_workers", False) and data_cfg.get("num_workers", 0) > 0),
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **loader_args)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_args)

    model = AudioFineTuneModel(model_cfg, task, num_class=num_class)
    latest_path = folder / "latest.pt"
    will_resume = bool(
        meta.get("load_checkpoint", True)
        and latest_path.is_file()
        and not meta.get("reset_epoch", False)
    )
    # On a fresh DDP run, rank 0 loads the large backbone and DDP broadcasts it.
    # Resume checkpoints include optimizer state, so every rank restores those below.
    if not will_resume and (not distributed or is_main):
        load_audio_backbone(model.backbone, meta.get("read_checkpoint"),
                            min_parameter_ratio=float(meta.get("min_load_ratio", 0.95)))
    frozen = bool(meta.get("frozen_encoder", False))
    if frozen:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    optimizer = _optimizer(model, opt_cfg, frozen)
    epochs = int(opt_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(opt_cfg.get("final_lr", 0.0))
    )
    start_epoch = 0
    higher_is_better = task != "regression"
    best_score = -math.inf if higher_is_better else math.inf
    if will_resume:
        checkpoint = _restore(latest_path, model, optimizer, scheduler)
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", best_score))

    if distributed:
        model = DistributedDataParallel(model, device_ids=[0] if device.type == "cuda" else None,
                                        find_unused_parameters=False)
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    mixed = device.type == "cuda" and dtype_name in {"bfloat16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype_name == "float16")

    if task == "classification":
        if not isinstance(num_class, int) or num_class < 2:
            raise ValueError("classification requires data.num_class >= 2")
        counts = Counter(train_dataset.labels)
        weights = torch.tensor([1 / max(1, counts[index]) for index in range(num_class)], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights / weights.sum() * num_class)
    elif task == "multi_label_classification":
        targets = np.asarray(train_dataset.labels)
        positives = targets.sum(axis=0)
        negatives = len(targets) - positives
        pos_weight = np.divide(negatives, positives, out=np.zeros_like(negatives), where=positives > 0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    elif task == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported task: {task}")

    history_path = folder / "history.csv"
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_result = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            task, True, mixed, amp_dtype, epoch=epoch, epochs=epochs, is_main=is_main,
        )
        val_result = _run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            task, False, mixed, amp_dtype, epoch=epoch, epochs=epochs, is_main=is_main,
        )
        scheduler.step()
        gathered = _gather((train_result, val_result), distributed, world_size)
        if is_main:
            train_loss = sum(item[0][0] for item in gathered) / max(1, sum(item[0][1] for item in gathered))
            val_loss = sum(item[1][0] for item in gathered) / max(1, sum(item[1][1] for item in gathered))
            train_truth = sum((item[0][3] for item in gathered), [])
            val_truth = sum((item[1][3] for item in gathered), [])
            val_paths = sum((item[1][2] for item in gathered), [])
            train_outputs = np.concatenate(sum((item[0][4] for item in gathered), []), axis=0)
            val_outputs = np.concatenate(sum((item[1][4] for item in gathered), []), axis=0)
            row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                   "backbone_lr": optimizer.param_groups[0]["lr"],
                   "head_lr": optimizer.param_groups[-1]["lr"]}
            if task == "classification":
                train_metrics, _, _, _ = classification_metrics(train_truth, train_outputs, num_class)
                val_metrics, _, predictions, probabilities = classification_metrics(val_truth, val_outputs, num_class)
                row.update({f"train_{key}": value for key, value in train_metrics.items() if not key.startswith("class_")})
                row.update({f"val_{key}": value for key, value in val_metrics.items() if not key.startswith("class_")})
                score = val_metrics["f1_macro"]
            elif task == "multi_label_classification":
                train_metrics, _, _ = multilabel_metrics(train_truth, train_outputs, num_class)
                val_metrics, predictions, probabilities = multilabel_metrics(val_truth, val_outputs, num_class)
                row.update({f"train_{key}": value for key, value in train_metrics.items() if not key.startswith("class_")})
                row.update({f"val_{key}": value for key, value in val_metrics.items() if not key.startswith("class_")})
                score = val_metrics["f1_macro"]
            else:
                predictions = val_outputs.squeeze(-1)
                targets = np.asarray(val_truth, dtype=np.float64)
                row["val_mae"] = float(np.abs(predictions - targets).mean())
                row["val_rmse"] = float(np.sqrt(np.square(predictions - targets).mean()))
                score = row["val_rmse"]
                probabilities = predictions
            improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score = score
                _checkpoint(folder / "best.pt", model, optimizer, scheduler, epoch + 1, best_score, args)
                with (folder / "best_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle); writer.writerow(["path", "target", "prediction", "output"])
                    writer.writerows(zip(val_paths, val_truth, np.asarray(predictions).tolist(), np.asarray(probabilities).tolist()))
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            _checkpoint(latest_path, model, optimizer, scheduler, epoch + 1, best_score, args)
            if task == "classification":
                LOGGER.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f "
                    "train_f1=%.4f val_f1=%.4f best=%.4f",
                    epoch + 1, train_loss, val_loss,
                    train_metrics["accuracy"], val_metrics["accuracy"],
                    train_metrics["f1_macro"], val_metrics["f1_macro"], best_score,
                )
            elif task == "multi_label_classification":
                LOGGER.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f train_subset_acc=%.4f "
                    "val_subset_acc=%.4f train_f1=%.4f val_f1=%.4f best=%.4f",
                    epoch + 1, train_loss, val_loss,
                    train_metrics["accuracy"], val_metrics["accuracy"],
                    train_metrics["f1_macro"], val_metrics["f1_macro"], best_score,
                )
            else:
                LOGGER.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f val_mae=%.4f "
                    "val_rmse=%.4f best=%.4f",
                    epoch + 1, train_loss, val_loss,
                    row["val_mae"], row["val_rmse"], best_score,
                )
        if distributed:
            dist.barrier()
    return {"latest_checkpoint": latest_path, "best_checkpoint": folder / "best.pt"}
