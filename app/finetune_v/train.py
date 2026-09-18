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
    multilabel_metrics,
    save_best_reports,
    save_metric_curves,
    save_multilabel_best_reports,
    write_history_csv,
    write_multilabel_predictions_csv,
    write_predictions_csv,
)
from utils.progress import progress_ncols


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


def _regression_extra_metrics(predictions, targets):
    """计算 MSE、R² 与调整后 R²（单输出回归，p=1）。"""
    n = len(targets)
    mse = float(np.square(predictions - targets).mean())
    ss_res = float(np.square(targets - predictions).sum())
    ss_tot = float(np.square(targets - targets.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if n > 2 else float("nan")
    return mse, r2, adjusted_r2


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
    frame_step = cfgs_data.get("frame_step")
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
        num_class=num_class,
        frames_per_clip=max_num_frames,
        fps=fps,
        frame_step=frame_step,
        train_transform=train_transform,
        val_transform=make_finetune_transforms(False, crop_size),
        num_clips=num_clips,
    )
    if is_main:
        LOGGER.info(
            "Loaded dataset: train=%d val=%d (task=%s, label_column=%s, num_class=%s)",
            len(train_dataset), len(val_dataset), task, label_column, num_class,
        )
        if task == "multi_label_classification":
            # Counts are per-class positive frequencies; Counter would reject the vectors.
            LOGGER.info(
                "Train positive counts=%s | val positive counts=%s",
                np.asarray(train_dataset.labels).sum(axis=0).astype(int).tolist(),
                np.asarray(val_dataset.labels).sum(axis=0).astype(int).tolist(),
            )
        else:
            LOGGER.info(
                "Train class counts=%s | val class counts=%s",
                dict(sorted(Counter(train_dataset.labels).items())),
                dict(sorted(Counter(val_dataset.labels).items())),
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
    if is_main:
        backbone_total = sum(p.numel() for p in encoder.parameters())
        backbone_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in task_head.parameters())
        head_trainable = sum(p.numel() for p in task_head.parameters() if p.requires_grad)
        LOGGER.info(
            "Model params: Total=%.4fM/%.4fM Backbone=%.4fM/%.4fM Head=%.4fM/%.4fM",
            (backbone_total + head_total) / 1e6, (backbone_trainable + head_trainable) / 1e6,
            backbone_total / 1e6, backbone_trainable / 1e6,
            head_total / 1e6, head_trainable / 1e6,
        )

    latest_path = folder / "latest.pt"
    higher_is_better = task in {"classification", "multi_label_classification"}
    start_epoch, best_score, history = 0, float("-inf") if higher_is_better else float("inf"), []
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
        # import math
        # counts = Counter(train_dataset.labels)
        # weights = torch.tensor(
        #     [ 1.0 / math.sqrt(max(1, counts[i]))   for i in range(num_class)     ],      device=device,
        # )
        # weights = weights / weights.sum() * num_class
        # criterion = nn.CrossEntropyLoss(weight=weights)
        
    elif task == "multi_label_classification":
        if not isinstance(num_class, int) or num_class < 2:
            raise ValueError("multi_label_classification requires data.num_class >= 2")
        targets = np.asarray(train_dataset.labels, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != num_class:
            raise ValueError(
                f"Expected multi-hot labels of shape [N, {num_class}], got {targets.shape}"
            )
        positives = targets.sum(axis=0)
        negatives = len(targets) - positives
        # Inverse-frequency weighting per class, mirroring the single-label CE weights.
        # Classes with no positive example get weight 0 so they cannot destabilize BCE.
        pos_weight = np.divide(negatives, positives, out=np.zeros_like(negatives), where=positives > 0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
        if is_main:
            absent = np.flatnonzero(positives == 0).tolist()
            LOGGER.info("pos_weight range=[%.2f, %.2f]; classes with no positive sample=%s",
                        float(pos_weight.min()), float(pos_weight.max()), absent)
    elif task == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(
            "data.task must be 'classification', 'multi_label_classification', or 'regression'"
        )

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
        for clips, labels, _, paths in tqdm(
            train_loader,
            disable=not is_main,
            desc=f"train {epoch + 1}/{num_epochs}",
            ncols=progress_ncols(),
        ):
            clips = _move_clips(clips, device)
            labels = labels.to(device, non_blocking=True)
            labels = labels.long() if task == "classification" else labels.float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
                outputs = _task_outputs(encoder, task_head, clips, frozen_encoder)
                loss = criterion(outputs if higher_is_better else outputs.squeeze(-1), labels)
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
                        loss = criterion(outputs if higher_is_better else outputs.squeeze(-1), labels)
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
                elif task == "multi_label_classification":
                    train_metrics, train_predictions, train_probabilities = multilabel_metrics(
                        train_truth, train_array, num_class)
                    val_metrics, val_predictions, val_probabilities = multilabel_metrics(
                        val_truth, val_array, num_class)
                    epoch_metrics.update({f"train_{key}": value for key, value in train_metrics.items() if not key.startswith("class_")})
                    epoch_metrics.update({f"val_{key}": value for key, value in val_metrics.items() if not key.startswith("class_")})
                    score = val_metrics["f1_macro"]
                    improved = score > best_score
                    if improved:
                        save_multilabel_best_reports(logs_dir, epoch + 1, val_metrics, val_paths, val_truth,
                                                     val_predictions, val_array, val_probabilities)
                        write_multilabel_predictions_csv(logs_dir / "train_best_predict.csv", train_paths, train_truth,
                                                         train_predictions, train_array, train_probabilities)
                        write_multilabel_predictions_csv(logs_dir / "eval_best_predict.csv", val_paths, val_truth,
                                                         val_predictions, val_array, val_probabilities)
                else:
                    train_predictions = train_array.squeeze(-1)
                    val_predictions = val_array.squeeze(-1)
                    train_targets = np.asarray(train_truth, dtype=np.float64)
                    val_targets = np.asarray(val_truth, dtype=np.float64)
                    train_mse, train_r2, train_adjusted_r2 = _regression_extra_metrics(train_predictions, train_targets)
                    val_mse, val_r2, val_adjusted_r2 = _regression_extra_metrics(val_predictions, val_targets)
                    epoch_metrics.update({
                        "train_mae": float(np.abs(train_predictions - train_targets).mean()),
                        "train_rmse": float(np.sqrt(np.square(train_predictions - train_targets).mean())),
                        "train_mse": train_mse,
                        "train_r2": train_r2,
                        "train_adjusted_r2": train_adjusted_r2,
                        "val_mae": float(np.abs(val_predictions - val_targets).mean()),
                        "val_rmse": float(np.sqrt(np.square(val_predictions - val_targets).mean())),
                        "val_mse": val_mse,
                        "val_r2": val_r2,
                        "val_adjusted_r2": val_adjusted_r2,
                    })
                    # 回归任务以 val_rmse 选 best（越低越好）
                    score = epoch_metrics["val_rmse"]
                    improved = score < best_score
                    if improved:
                        _write_regression_best_logs(logs_dir, epoch + 1, train_paths, train_truth, train_predictions,
                                                     val_paths, val_truth, val_predictions,
                                                     {"val_loss": epoch_metrics["val_loss"], "val_mse": val_mse,
                                                      "val_mae": epoch_metrics["val_mae"], "val_rmse": epoch_metrics["val_rmse"],
                                                      "val_r2": val_r2, "val_adjusted_r2": val_adjusted_r2})
                history.append(epoch_metrics)
                write_history_csv(history, logs_dir / "history.csv")
                save_metric_curves(history, logs_dir / "metrics_curves.png")
                if improved:
                    best_score = score
                    _save_checkpoint(folder / "best.pt", epoch + 1, encoder, task_head, optimizer, scaler,
                                     scheduler, wd_scheduler, best_score, history, args)
                if task == "classification":
                    LOGGER.info(
                        "epoch=%d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f train_f1=%.4f val_f1=%.4f best=%.4f",
                        epoch + 1,
                        epoch_metrics["train_loss"], epoch_metrics["val_loss"],
                        epoch_metrics["train_accuracy"], epoch_metrics["val_accuracy"],
                        epoch_metrics["train_f1_macro"], epoch_metrics["val_f1_macro"],
                        best_score,
                    )
                elif task == "multi_label_classification":
                    LOGGER.info(
                        "epoch=%d train_loss=%.4f val_loss=%.4f train_subset_acc=%.4f val_subset_acc=%.4f "
                        "train_f1=%.4f val_f1=%.4f val_hamming=%.4f best=%.4f",
                        epoch + 1,
                        epoch_metrics["train_loss"], epoch_metrics["val_loss"],
                        epoch_metrics["train_accuracy"], epoch_metrics["val_accuracy"],
                        epoch_metrics["train_f1_macro"], epoch_metrics["val_f1_macro"],
                        epoch_metrics["val_hamming_loss"],
                        best_score,
                    )
                else:
                    LOGGER.info(
                        "epoch=%d train_loss=%.4f val_loss=%.4f train_mae=%.4f val_mae=%.4f train_rmse=%.4f val_rmse=%.4f best=%.4f",
                        epoch + 1,
                        epoch_metrics["train_loss"], epoch_metrics["val_loss"],
                        epoch_metrics["train_mae"], epoch_metrics["val_mae"],
                        epoch_metrics["train_rmse"], epoch_metrics["val_rmse"],
                        best_score,
                    )

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
