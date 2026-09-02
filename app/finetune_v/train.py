"""Distributed video fine-tuning entry point."""

from __future__ import annotations

import contextlib
import csv
import logging
import os
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

from datasets.video_finetune_dataset import make_videodataset_finetune_v
from datasets.video_transforms import make_finetune_transforms
from models.finetune_v_model import build_model
from optimization.optimizer import init_ft_opt
from utils.classification_metrics import (
    classification_metrics,
    history_from_csv,
    save_best_reports,
    save_metric_curves,
    write_history_csv,
    write_predictions_csv,
)


LOGGER = logging.getLogger(__name__)


class DistributedEvaluationSampler(Sampler):
    """Partition evaluation samples across ranks without padding duplicates."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def _distributed_state():
    enabled = dist.is_available() and dist.is_initialized()
    return enabled, dist.get_rank() if enabled else 0, dist.get_world_size() if enabled else 1


def _unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _move_clips(clips, device):
    return [[view.to(device, non_blocking=True) for view in clip] for clip in clips]


def _task_outputs(encoder, task_head, clips, frozen_encoder):
    """Encode all clips once, then average one task-head prediction per clip."""
    encoder_context = torch.no_grad() if frozen_encoder else contextlib.nullcontext()
    with encoder_context:
        encoded_clips = encoder(clips)
    batch_size = encoded_clips[0].shape[0]
    tokens = torch.cat(encoded_clips, dim=0)
    outputs = task_head(tokens)
    return outputs.reshape(len(encoded_clips), batch_size, -1).mean(dim=0)


def _gather_object(value, world_size, distributed):
    if not distributed:
        return value
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _restore_checkpoint(path, encoder, task_head, optimizer, scaler, scheduler, wd_scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(encoder).load_state_dict(checkpoint["encoder"])
    _unwrap(task_head).load_state_dict(checkpoint["task_head"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    scheduler._step = checkpoint.get("scheduler_step", 0.0)
    wd_scheduler._step = checkpoint.get("wd_scheduler_step", 0.0)
    return checkpoint


def _save_checkpoint(path, epoch, encoder, task_head, optimizer, scaler, scheduler, wd_scheduler,
                     best_score, history, args):
    state = {
        "epoch": epoch,
        "encoder": _unwrap(encoder).state_dict(),
        "task_head": _unwrap(task_head).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "scheduler_step": scheduler._step,
        "wd_scheduler_step": wd_scheduler._step,
        "best_score": best_score,
        "history": history,
        "args": args,
    }
    torch.save(state, path)


def _write_regression_best_logs(logs_dir, epoch, train_paths, train_truth, train_predictions,
                                eval_paths, eval_truth, eval_predictions, metrics):
    def write_predictions(path, paths, truth, predictions):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "target", "prediction", "error"])
            writer.writeheader()
            for sample_path, target, prediction in zip(paths, truth, predictions):
                writer.writerow({
                    "path": sample_path,
                    "target": float(target),
                    "prediction": float(prediction),
                    "error": float(prediction - target),
                })

    logs_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(logs_dir / "train_best_predict.csv", train_paths, train_truth, train_predictions)
    write_predictions(logs_dir / "eval_best_predict.csv", eval_paths, eval_truth, eval_predictions)
    with (logs_dir / "best.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *metrics])
        writer.writeheader()
        writer.writerow({"epoch": epoch, **metrics})


def main(args):
    distributed, rank, world_size = _distributed_state()
    is_main = rank == 0
    cfgs_meta = args.get("meta", {})
    seed = cfgs_meta.get("seed", 0)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    folder = Path(args["folder"])
    logs_dir = folder / "logs"
    if is_main:
        folder.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = folder / "train.log"
        if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
                   for handler in LOGGER.handlers):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization", {})
    num_epochs = cfgs_opt["epochs"]
    final_wd = float(cfgs_opt["final_weight_decay"])
    wd = float(cfgs_opt["weight_decay"])
    warmup = cfgs_opt["warmup"]
    start_lr = cfgs_opt["start_lr"]
    lr = cfgs_opt["lr"]
    final_lr = cfgs_opt["final_lr"]
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    frozen_encoder = cfgs_meta.get("frozen_encoder", False)

    # -- MODEL
    cfgs_model = args.get("model", {})
    model_name = cfgs_model["model_name"]
    uniform_power = cfgs_model.get("uniform_power", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_sdpa = cfgs_model.get("use_sdpa", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    out_layers = cfgs_model.get("out_layers")
    classifier_depth = cfgs_model.get("classifier_depth", 1)
    classifier_num_heads = cfgs_model.get("classifier_num_heads")

    # -- DATA
    cfgs_data = args.get("data", {})
    task = cfgs_data.get("task", "classification").lower()
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size", 16)
    dataset_fpcs = cfgs_data["dataset_fpcs"]
    max_num_frames = max(dataset_fpcs)
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    num_class = cfgs_data.get("num_class")
    label_column = cfgs_data["label_column"]
    dataset_paths = cfgs_data["datasets"]
    root_paths = cfgs_data["rootpaths"]
    fps = cfgs_data.get("fps")
    num_clips = cfgs_data.get("num_clips", 1)
    batch_size = cfgs_data["batch_size"]
    num_workers = cfgs_data.get("num_workers", 0)
    pin_mem = cfgs_data.get("pin_mem", False)
    persistent_workers = cfgs_data.get("persistent_workers", False)

    # -- DATA AUGMENTATION
    cfgs_data_aug = args.get("data_aug", {})
    train_transform = make_finetune_transforms(
        True,
        crop_size,
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [0.85, 1.15]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.6, 1.0]),
        random_horizontal_flip=cfgs_data_aug.get("random_horizontal_flip", 0.5),
        reprob=cfgs_data_aug.get("reprob", 0.0),
    )
    train_dataset, val_dataset = make_videodataset_finetune_v(
        dataset_paths,
        label_column=label_column,
        task=task,
        root_paths=root_paths,
        frames_per_clip=max_num_frames,
        fps=fps,
        train_transform=train_transform,
        val_transform=make_finetune_transforms(False, crop_size),
        num_clips=num_clips,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedEvaluationSampler(val_dataset, rank, world_size) if distributed else None
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs)
    if not len(train_loader):
        raise ValueError("Training DataLoader is empty; reduce data.batch_size or add training samples")
    iterations_per_epoch = len(train_loader)

    # -- MODEL / OPTIMIZER
    pretrained_ckpt = cfgs_meta.get("read_checkpoint")
    which_dtype = cfgs_meta.get("dtype", "float32").lower()
    mixed_precision = device.type == "cuda" and which_dtype in {"bfloat16", "float16"}
    amp_dtype = torch.bfloat16 if which_dtype == "bfloat16" else torch.float16
    encoder, task_head = build_model(
        pretrained_ckpt,
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        task=task,
        num_class=num_class,
        out_layers=out_layers,
        classifier_depth=classifier_depth,
        classifier_num_heads=classifier_num_heads,
        uniform_power=uniform_power,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
    )
    encoder.to(device)
    task_head.to(device)
    optimizer, scaler, scheduler, wd_scheduler = init_ft_opt(
        encoder=encoder,
        predictor=task_head,
        iterations_per_epoch=iterations_per_epoch,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        frozen_encoder=frozen_encoder,
    )

    latest_path = folder / "latest.pt"
    start_epoch, best_score, history = 0, float("-inf") if task == "classification" else float("inf"), []
    if cfgs_meta.get("load_checkpoint", True) and latest_path.is_file() and not cfgs_meta.get("reset_epoch", False):
        checkpoint = _restore_checkpoint(latest_path, encoder, task_head, optimizer, scaler, scheduler, wd_scheduler)
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", best_score))
        history = checkpoint.get("history", [])
        if is_main:
            LOGGER.info("Resumed fine-tuning checkpoint %s at epoch %d", latest_path, start_epoch)
    elif (logs_dir / "history.csv").is_file():
        try:
            history = history_from_csv(logs_dir / "history.csv")
        except (ValueError, OSError):
            history = []

    if distributed and not frozen_encoder:
        encoder = DistributedDataParallel(encoder, device_ids=[0] if device.type == "cuda" else None, find_unused_parameters=False)
    if distributed:
        task_head = DistributedDataParallel(task_head, device_ids=[0] if device.type == "cuda" else None, find_unused_parameters=False)

    if task == "classification":
        if not isinstance(num_class, int) or num_class < 2:
            raise ValueError("Classification requires data.num_class >= 2")
        invalid_labels = sorted({label for label in train_dataset.labels + val_dataset.labels
                                 if label < 0 or label >= num_class})
        if invalid_labels:
            raise ValueError(
                f"Classification labels must be in [0, {num_class - 1}], "
                f"but found {invalid_labels}. CSV {label_column!r} labels must be in [1, {num_class}]."
            )
        counts = Counter(train_dataset.labels)
        weights = torch.tensor([1.0 / max(1, counts[index]) for index in range(num_class)], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights / weights.sum() * num_class)
    elif task == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError("data.task must be 'classification' or 'regression'")

    eval_freq = max(1, int(cfgs_meta.get("eval_freq", 1)))
    save_every_freq = int(cfgs_meta.get("save_every_freq", -1))
    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if frozen_encoder:
            encoder.eval()
        else:
            encoder.train()
        task_head.train()
        train_stats = torch.zeros(2, dtype=torch.float64, device=device)
        train_paths, train_truth, train_outputs = [], [], []
        for clips, labels, _, paths in tqdm(train_loader, disable=not is_main, desc=f"train {epoch + 1}/{num_epochs}"):
            clips = _move_clips(clips, device)
            labels = labels.to(device, non_blocking=True)
            labels = labels.long() if task == "classification" else labels.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                outputs = _task_outputs(encoder, task_head, clips, frozen_encoder)
                loss = criterion(outputs if task == "classification" else outputs.squeeze(-1), labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            wd_scheduler.step()
            train_stats += torch.tensor([loss.detach().item() * len(labels), len(labels)], dtype=torch.float64, device=device)
            train_paths.extend(paths)
            train_truth.extend(labels.detach().cpu().tolist())
            train_outputs.append(outputs.detach().float().cpu().numpy())

        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        should_evaluate = (epoch + 1) % eval_freq == 0 or epoch + 1 == num_epochs
        if should_evaluate:
            encoder.eval()
            task_head.eval()
            val_stats = torch.zeros(2, dtype=torch.float64, device=device)
            val_paths, val_truth, val_outputs = [], [], []
            with torch.no_grad():
                for clips, labels, _, paths in val_loader:
                    clips = _move_clips(clips, device)
                    labels = labels.to(device, non_blocking=True)
                    labels = labels.long() if task == "classification" else labels.float()
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                        outputs = _task_outputs(encoder, task_head, clips, frozen_encoder)
                        loss = criterion(outputs if task == "classification" else outputs.squeeze(-1), labels)
                    val_stats += torch.tensor([loss.item() * len(labels), len(labels)], dtype=torch.float64, device=device)
                    val_paths.extend(paths)
                    val_truth.extend(labels.cpu().tolist())
                    val_outputs.append(outputs.float().cpu().numpy())
            if distributed:
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
                gathered = _gather_object(
                    (train_paths, train_truth, train_outputs, val_paths, val_truth, val_outputs), world_size, True
                )
                train_paths = sum((item[0] for item in gathered), [])
                train_truth = sum((item[1] for item in gathered), [])
                train_outputs = sum((item[2] for item in gathered), [])
                val_paths = sum((item[3] for item in gathered), [])
                val_truth = sum((item[4] for item in gathered), [])
                val_outputs = sum((item[5] for item in gathered), [])

            if is_main:
                train_array = np.concatenate(train_outputs, axis=0)
                val_array = np.concatenate(val_outputs, axis=0)
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_stats[0].item() / train_stats[1].item()),
                    "val_loss": float(val_stats[0].item() / val_stats[1].item()),
                    "train_samples": len(train_truth),
                    "val_samples": len(val_truth),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
                }
                if task == "classification":
                    train_metrics, _, train_predictions, train_probabilities = classification_metrics(train_truth, train_array, num_class)
                    val_metrics, matrix, val_predictions, val_probabilities = classification_metrics(val_truth, val_array, num_class)
                    epoch_metrics.update({f"train_{key}": value for key, value in train_metrics.items() if not key.startswith("class_")})
                    epoch_metrics.update({f"val_{key}": value for key, value in val_metrics.items() if not key.startswith("class_")})
                    epoch_metrics.update({f"cm_{row}_{col}": int(matrix[row, col]) for row in range(num_class) for col in range(num_class)})
                    score = val_metrics["f1_macro"]
                    improved = score > best_score
                    if improved:
                        save_best_reports(logs_dir, epoch + 1, val_metrics, matrix, val_paths, val_truth,
                                          val_predictions, val_array, val_probabilities)
                        write_predictions_csv(logs_dir / "train_best_predict.csv", train_paths, train_truth,
                                              train_predictions, train_array, train_probabilities)
                        write_predictions_csv(logs_dir / "eval_best_predict.csv", val_paths, val_truth,
                                              val_predictions, val_array, val_probabilities)
                else:
                    train_predictions = train_array.squeeze(-1)
                    val_predictions = val_array.squeeze(-1)
                    train_targets = np.asarray(train_truth, dtype=np.float64)
                    val_targets = np.asarray(val_truth, dtype=np.float64)
                    epoch_metrics.update({
                        "train_mae": float(np.abs(train_predictions - train_targets).mean()),
                        "train_rmse": float(np.sqrt(np.square(train_predictions - train_targets).mean())),
                        "val_mae": float(np.abs(val_predictions - val_targets).mean()),
                        "val_rmse": float(np.sqrt(np.square(val_predictions - val_targets).mean())),
                    })
                    score = epoch_metrics["val_loss"]
                    improved = score < best_score
                    if improved:
                        _write_regression_best_logs(logs_dir, epoch + 1, train_paths, train_truth, train_predictions,
                                                     val_paths, val_truth, val_predictions,
                                                     {"val_loss": epoch_metrics["val_loss"], "val_mae": epoch_metrics["val_mae"], "val_rmse": epoch_metrics["val_rmse"]})
                history.append(epoch_metrics)
                write_history_csv(history, logs_dir / "history.csv")
                save_metric_curves(history, logs_dir / "metrics_curves.png")
                if improved:
                    best_score = score
                    _save_checkpoint(folder / "best.pt", epoch + 1, encoder, task_head, optimizer, scaler,
                                     scheduler, wd_scheduler, best_score, history, args)
                LOGGER.info("epoch=%d train_loss=%.5f val_loss=%.5f best=%s", epoch + 1,
                            epoch_metrics["train_loss"], epoch_metrics["val_loss"], best_score)

        if is_main:
            _save_checkpoint(latest_path, epoch + 1, encoder, task_head, optimizer, scaler, scheduler,
                             wd_scheduler, best_score, history, args)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                _save_checkpoint(folder / f"e{epoch + 1}.pt", epoch + 1, encoder, task_head,
                                 optimizer, scaler, scheduler, wd_scheduler, best_score, history, args)
        if distributed:
            dist.barrier()

    return {
        "latest_checkpoint": latest_path,
        "best_checkpoint": folder / "best.pt",
        "logs_dir": logs_dir,
    }
