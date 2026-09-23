import argparse
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.progress import progress_ncols

from models.finetune_v_model import build_model

from datasets.video_jepa.finetune_dataset import VideoCSVDataset
from datasets.video_jepa.transforms import make_finetune_transforms

from utils.classification_metrics import (classification_metrics, save_best_reports, save_metric_curves,
                   write_history_csv, write_predictions_csv)


def args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained V-JEPA ViT-L using a YAML configuration"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    config_path = parser.parse_args().config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    config["config_path"] = config_path
    return argparse.Namespace(**config)


def setup(seed):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
    random.seed(seed + local_rank); np.random.seed(seed + local_rank); torch.manual_seed(seed + local_rank)
    return distributed, local_rank, dist.get_world_size() if distributed else 1


def reduce_metrics(values, device, distributed):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed: dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


def main():
    opt = args(); distributed, rank, world = setup(opt.seed)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        Path(opt.output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(opt.config_path, Path(opt.output_dir) / opt.config_path.name)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(Path(opt.output_dir) / "train.log")])
    log = logging.getLogger(__name__)
    train_ds = VideoCSVDataset(opt.csv_path, opt.train_split, opt.frames_per_clip, opt.frame_step,
                               make_transforms(True, opt.crop_size), True, opt.num_clips)
    val_ds = VideoCSVDataset(opt.csv_path, opt.val_split, opt.frames_per_clip, opt.frame_step,
                             make_transforms(False, opt.crop_size), False, opt.num_clips)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    loader_kw = dict(batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True,
                     persistent_workers=opt.num_workers > 0)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None, **loader_kw)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **loader_kw)
    
    
    
    encoder, classifier = build_model(opt.pretrained_ckpt, opt.crop_size, opt.frames_per_clip, opt.num_classes)
    encoder.to(device); classifier.to(device)
    if opt.freeze_backbone:
        for parameter in encoder.parameters(): parameter.requires_grad = False
    if distributed:
        encoder = DDP(encoder, device_ids=[rank], find_unused_parameters=False)
        classifier = DDP(classifier, device_ids=[rank], find_unused_parameters=False)


        
    counts = Counter(train_ds.labels)
    weights = torch.tensor([1.0 / max(1, counts[i]) for i in range(opt.num_classes)], device=device)
    weights = weights / weights.sum() * opt.num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    
    params = list(classifier.parameters()) + ([] if opt.freeze_backbone else list(encoder.parameters()))
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16 if opt.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    
    
    best = -1.0
    history = []
    output_dir = Path(opt.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(opt.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        encoder.train(not opt.freeze_backbone); classifier.train()
        train_stats = np.zeros(3)
        train_truth, train_paths, train_logits = [], [], []
        for clips, labels, _, paths in tqdm(
            train_loader,
            disable=rank != 0,
            desc=f"train {epoch + 1}/{opt.epochs}",
            ncols=progress_ncols(),
        ):
            labels = labels.to(device, non_blocking=True)
            clips = [[x.to(device, non_blocking=True) for x in group] for group in clips]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = torch.stack([classifier(x) for x in encoder(clips)]).mean(0)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_stats += [loss.item() * len(labels), (logits.argmax(1) == labels).sum().item(), len(labels)]
            train_truth.extend(labels.detach().cpu().tolist())
            train_paths.extend(paths)
            train_logits.append(logits.detach().float().cpu().numpy())
        train_stats = reduce_metrics(train_stats, device, distributed)
        encoder.eval(); classifier.eval(); val_stats = np.zeros(3)
        preds, truth, val_paths, val_logits = [], [], [], []
        with torch.no_grad():
            for clips, labels, _, paths in val_loader:
                labels = labels.to(device); clips = [[x.to(device) for x in group] for group in clips]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = torch.stack([classifier(x) for x in encoder(clips)]).mean(0); loss = criterion(logits, labels)
                pred = logits.argmax(1); val_stats += [loss.item() * len(labels), (pred == labels).sum().item(), len(labels)]
                preds.extend(pred.cpu().tolist()); truth.extend(labels.cpu().tolist())
                val_paths.extend(paths)
                val_logits.append(logits.float().cpu().numpy())
        val_stats = reduce_metrics(val_stats, device, distributed)
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, (
                train_truth, train_paths, train_logits, preds, truth, val_paths, val_logits
            ))
            train_truth = sum((item[0] for item in gathered), [])
            train_paths = sum((item[1] for item in gathered), [])
            train_logits = sum((item[2] for item in gathered), [])
            preds = sum((item[3] for item in gathered), [])
            truth = sum((item[4] for item in gathered), [])
            val_paths = sum((item[5] for item in gathered), [])
            val_logits = sum((item[6] for item in gathered), [])
        if rank == 0:
            train_logits_array = np.concatenate(train_logits, axis=0)
            logits_array = np.concatenate(val_logits, axis=0)
            train_metric_values, _, train_preds_array, train_probabilities = classification_metrics(
                train_truth, train_logits_array, opt.num_classes
            )
            metric_values, matrix, preds_array, probabilities = classification_metrics(
                truth, logits_array, opt.num_classes
            )
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_stats[0] / train_stats[2]),
                "train_acc": float(100 * train_stats[1] / train_stats[2]),
                "val_loss": float(val_stats[0] / val_stats[2]),
                "val_acc": float(100 * val_stats[1] / val_stats[2]),
                "train_samples": len(train_truth),
                "val_samples": len(truth),
                **{f"train_{key}": value for key, value in train_metric_values.items()
                   if not key.startswith("class_")},
                **{f"val_{key}": value for key, value in metric_values.items()
                   if not key.startswith("class_")},
            }
            epoch_metrics.update({
                f"cm_{row}_{column}": int(matrix[row, column])
                for row in range(matrix.shape[0])
                for column in range(matrix.shape[1])
            })
            history.append(epoch_metrics)
            write_history_csv(history, logs_dir / "history.csv")
            log.info("epoch=%d train_loss=%.4f train_acc=%.2f val_loss=%.4f val_acc=%.2f balanced_acc=%.4f f1=%.4f\n%s",
                     epoch + 1, epoch_metrics["train_loss"], epoch_metrics["train_acc"],
                     epoch_metrics["val_loss"], epoch_metrics["val_acc"],
                     metric_values["balanced_accuracy"], metric_values["f1_macro"], matrix)
            state = {"encoder": (encoder.module if distributed else encoder).backbone.state_dict(),
                     "classifiers": [(classifier.module if distributed else classifier).state_dict()],
                     "epoch": epoch, "args": vars(opt), "balanced_accuracy": metric_values["balanced_accuracy"],
                     "f1": metric_values["f1_macro"]}
            torch.save(state, output_dir / "latest.pt")
            if metric_values["f1_macro"] > best:
                best = metric_values["f1_macro"]
                torch.save(state, output_dir / "best.pt")
                # best.csv follows the compact sample.csv layout and reports
                # classification metrics on the validation samples.
                save_best_reports(logs_dir, epoch + 1, metric_values, matrix)
                write_predictions_csv(logs_dir / "train_best_predict.csv", train_paths,
                                      train_truth, train_preds_array, train_logits_array,
                                      train_probabilities)
                write_predictions_csv(logs_dir / "eval_best_predict.csv", val_paths,
                                      truth, preds_array, logits_array, probabilities)
    if rank == 0:
        save_metric_curves(history, logs_dir / "metrics_curves.png")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__": main()
