"""Paired VA downstream training; shared metrics and task-label conventions."""
import csv
import logging
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from app.pretrain_audio_lejepa.train import _state, _unwrap, _schedule, _append_csv
from app.pretrain_va_lejepa.train import _optimizer, _save_all, _restore_rng, _grad_norm, _record_skipped
from app.finetune_a.train import _criterion, _regression_metrics
from datasets.va_lejepa.finetune_dataset import AVCSVDataset
from datasets.va_lejepa.sync_preflight import prepare_sync_metadata, sync_preflight_path
from datasets.va_lejepa import collate_va_lejepa, to_device, synchronize_training_batch
from models.va_lejepa import build_va_lejepa, load_checkpoint
from models.va_lejepa.finetune import VAFinetuner
from utils.classification_metrics import (
    classification_metrics, multilabel_metrics, save_metric_curves, write_history_csv,
    save_best_reports, save_multilabel_best_reports, write_predictions_csv,
    write_multilabel_predictions_csv, write_regression_best_reports,
)
from datasets.common.instances import as_sentence_instances
from datasets.common.instances import sentence_identity
from utils.instance_evaluation import predict_instance_batch
from utils.progress import progress_ncols

LOGGER = logging.getLogger(__name__)


def _configure_file_logging(folder, rank):
    log_path = (folder / ('train.log' if rank == 0 else f'train-rank{rank}.log')).resolve()
    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_path:
                break
            LOGGER.removeHandler(handler)
            handler.close()
    else:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = rank == 0


class EvaluationSampler(Sampler):
    """Evaluate each sample exactly once, without DistributedSampler padding."""
    def __init__(self, dataset, rank, world):
        self.indices = list(range(rank, len(dataset), world))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)


def task_metrics(truth, logits, task, classes):
    if task == 'classification':
        metrics, matrix, predictions, probabilities = classification_metrics(truth, logits, classes)
        return metrics, predictions, probabilities, matrix
    if task == 'multi_label_classification':
        metrics, predictions, probabilities = multilabel_metrics(truth, logits, classes)
        return metrics, predictions, probabilities, None
    predictions = np.asarray(logits).reshape(-1)
    metrics = _regression_metrics(predictions, truth)
    if np.std(predictions) > 0 and np.std(truth) > 0:
        metrics['pearson'] = float(np.corrcoef(predictions, truth)[0, 1])
    else:
        metrics['pearson'] = float('nan')
    return metrics, predictions, predictions, None


def _loss(output, labels, criterion, task, branch_weight=0.0):
    def compute(logits):
        return criterion(logits.squeeze(-1) if task == 'regression' else logits, labels)
    main = compute(output['logits'])
    # Transformer aggregation uses branch supervision to make per-branch metrics meaningful.
    if branch_weight:
        main = main + branch_weight * sum(compute(logits) for logits in output['branches'].values()) / len(output['branches'])
    return main


@torch.no_grad()
def evaluate(model, loader, device, task, classes, epoch_number=None, total_epochs=None,
             criterion=None, branch_weight=0.0):
    model.eval()
    raw = _unwrap(model)
    records = []
    skipped = 0
    loss_total = torch.zeros(2, dtype=torch.float64, device=device)
    enabled, rank, world = _state()
    description = f'val {epoch_number}/{total_epochs}' if epoch_number is not None else 'val'
    for batch in tqdm(loader, disable=rank != 0, desc=description, ncols=progress_ncols()):
        skipped += len(batch.get('skipped_samples', []))
        if 'instance_samples' in batch:
            if not batch['instance_samples']:
                continue
            output, labels, identities = predict_instance_batch(batch, lambda window: raw(to_device(window, device)), task)
            if criterion is not None:
                loss_total += torch.tensor([float(_loss(output, labels.to(device), criterion, task, branch_weight)) * len(labels),
                                            len(labels)], dtype=torch.float64, device=device)
            for i, (sample, identity) in enumerate(zip(batch['instance_samples'], identities)):
                first = sample['sentences'][0]
                records.append(dict(pair_id=identity, video_path=identity, audio_path=sentence_identity(first['audio_path'])[0],
                    truth=labels[i].tolist(), logits=output['logits'][i].cpu().tolist(),
                    branches={mode: value[i].cpu().tolist() for mode, value in output['branches'].items()}))
            continue
        if 'global' not in batch:
            continue
        output = raw(to_device(batch, device))
        if criterion is not None:
            labels = batch['label'].to(device)
            loss_total += torch.tensor([float(_loss(output, labels, criterion, task, branch_weight)) * len(labels),
                                        len(labels)], dtype=torch.float64, device=device)
        for i, pair_id in enumerate(batch['pair_id']):
            records.append(dict(pair_id=pair_id, video_path=batch['video_path'][i], audio_path=batch['audio_path'][i],
                truth=batch['label'][i].cpu().tolist(), logits=output['logits'][i].float().cpu().tolist(),
                branches={mode: value[i].float().cpu().tolist() for mode, value in output['branches'].items()}))
    if skipped:
        LOGGER.warning('VA evaluation rank=%d skipped %d invalid pairs', rank, skipped)
    if enabled:
        dist.all_reduce(loss_total)
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [r for shard in gathered for r in shard]
    if not records:
        raise ValueError('empty validation split')
    metrics, predictions, probabilities, matrix = task_metrics([r['truth'] for r in records], [r['logits'] for r in records], task, classes)
    for mode in raw.va.enabled_modes:
        branch_metrics, _, _, _ = task_metrics([r['truth'] for r in records], [r['branches'][mode] for r in records], task, classes)
        metrics.update({f'{mode}_{key}': value for key, value in branch_metrics.items()})
    val_loss = float(loss_total[0] / loss_total[1]) if criterion is not None and loss_total[1] else float('nan')
    return metrics, records, predictions, probabilities, matrix, val_loss


def _epoch_summary(epoch, task, row, best):
    fields = [f'epoch={epoch}', f'train_loss={row["train_loss"]:.4f}', f'val_loss={row["val_loss"]:.4f}']
    if task == 'regression':
        for key in ('mae', 'rmse'):
            fields.extend((f'train_{key}={row[f"train_{key}"]:.4f}', f'val_{key}={row[f"val_{key}"]:.4f}'))
    elif task == 'multi_label_classification':
        fields.extend((f'train_subset_acc={row["train_accuracy"]:.4f}',
                       f'val_subset_acc={row["val_accuracy"]:.4f}',
                       f'train_f1={row["train_f1_macro"]:.4f}',
                       f'val_f1={row["val_f1_macro"]:.4f}',
                       f'val_hamming={row["val_hamming_loss"]:.4f}'))
    else:
        for key, name in (('accuracy', 'acc'), ('f1_macro', 'f1')):
            fields.extend((f'train_{name}={row[f"train_{key}"]:.4f}', f'val_{name}={row[f"val_{key}"]:.4f}'))
    fields.append(f'best={best:.4f}')
    return ' '.join(fields)


def _write_best_reports(logs_dir, epoch, task, classes, train_paths, train_truth, train_logits,
                        train_predictions, train_probabilities, records, metrics, predictions,
                        probabilities, matrix, val_loss, enabled_modes):
    val_paths = [record['video_path'] for record in records]
    val_truth = [record['truth'] for record in records]
    val_logits = np.asarray([record['logits'] for record in records])
    base_metrics = {key: value for key, value in metrics.items()
                    if not any(key.startswith(f'{mode}_') for mode in enabled_modes)}
    if task == 'classification':
        save_best_reports(logs_dir, epoch, base_metrics, matrix)
        write_predictions_csv(logs_dir / 'train_best_predict.csv', train_paths, train_truth,
                              train_predictions, train_logits, train_probabilities)
        write_predictions_csv(logs_dir / 'eval_best_predict.csv', val_paths, val_truth,
                              predictions, val_logits, probabilities)
    elif task == 'multi_label_classification':
        save_multilabel_best_reports(logs_dir, epoch, base_metrics, classes)
        write_multilabel_predictions_csv(logs_dir / 'train_best_predict.csv', train_paths, train_truth,
                                         train_predictions, train_logits, train_probabilities)
        write_multilabel_predictions_csv(logs_dir / 'eval_best_predict.csv', val_paths, val_truth,
                                         predictions, val_logits, probabilities)
    else:
        write_regression_best_reports(
            logs_dir, epoch, train_paths, train_truth, train_predictions,
            val_paths, val_truth, predictions,
            {key: val_loss if key == 'val_loss' else base_metrics[key.removeprefix('val_')]
             for key in ('val_loss', 'val_mse', 'val_mae', 'val_rmse', 'val_r2', 'val_adjusted_r2')})


def _initialize_va(model_cfg, data_cfg, cfg, resume=None, rank=0):
    path = cfg.get('pretrained_checkpoint')
    va = build_va_lejepa(model_cfg, data_cfg, initialize=not resume and not path and rank == 0)
    if not resume and path:
        load_checkpoint(path, va, allow_new_branches=True)
    elif not resume and rank == 0:
        LOGGER.info('VA finetune initialized from separate backbone checkpoints; fusion and task layers are newly initialized')
    return va


def main(args):
    distributed, rank, world = _state()
    data, cfg, opt = dict(args['data']), dict(args.get('finetune', {})), dict(args['optimization'])
    if 'instance_aggregator' in args['model']:
        cfg['instance_aggregator'] = args['model']['instance_aggregator']
    if int(data.get('num_sentence', 0)) > 0 and 'instance_aggregator' not in cfg:
        raise ValueError('data.num_sentence > 0 requires model.instance_aggregator')
    meta = args.get('meta', {})
    data['seed'] = int(meta.get('seed', 0))
    task = data.get('task', 'classification')
    classes = int(data.get('num_class', 2))
    seed = int(meta.get('seed', 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    folder = Path(args['folder'])
    (folder / 'logs').mkdir(parents=True, exist_ok=True)
    _configure_file_logging(folder, rank)
    resume = meta.get('read_checkpoint') or (folder / 'latest.pt' if meta.get('auto_resume', True) and (folder / 'latest.pt').exists() else None)
    LOGGER.info('VA finetune rank=%d/%d task=%s output=%s resume=%s pretrained_checkpoint=%s',
                rank, world, task, folder, resume, cfg.get('pretrained_checkpoint'))
    va = _initialize_va(args['model'], data, cfg, resume, rank)
    model = VAFinetuner(va, 1 if task == 'regression' else classes, cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[0] if device.type == 'cuda' else None, find_unused_parameters=True)
    raw = _unwrap(model)
    base_lr = float(opt.get('lr', opt.get('lr_head', .001)))
    for family, scale in (('video_pretrained', 'video'), ('audio_pretrained', 'audio'), ('new', 'fusion'), ('head', 'head')):
        opt['lr_' + family] = base_lr * float(cfg.get(scale + '_lr_scale', 1))
    optimizer = _optimizer(raw, opt)
    mixed = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    train_dataset, val_dataset = AVCSVDataset(data, 0, True), AVCSVDataset(data, 1, False)
    count = prepare_sync_metadata((train_dataset, val_dataset), data,
        sync_preflight_path(data, folder), rank=rank,
        workers=int(data.get('sync_preflight_workers', 8)))
    if rank == 0:
        LOGGER.info('VA sync preflight checked %d train/validation pairs', count)
    train_dataset = as_sentence_instances(train_dataset, data, 'audio-video', True)
    val_dataset = as_sentence_instances(val_dataset, data, 'audio-video', False)
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True)
    def loader(dataset, sample):
        return DataLoader(dataset, sampler=sample, batch_size=int(data.get('batch_size', 1)),
            num_workers=int(data.get('num_workers', 4)),
            collate_fn=getattr(dataset, 'collate_fn', collate_va_lejepa), pin_memory=True)
    train_loader = loader(train_dataset, sampler)
    val_loader = loader(val_dataset, EvaluationSampler(val_dataset, rank, world))
    if not len(train_loader):
        raise ValueError('empty train loader')
    # Same weighting/loss options as FAVOR audio finetuning.
    criterion = _criterion(task, opt, train_dataset, classes, device)
    branch_weight = float(cfg.get('branch_loss_weight', .1 if raw.strategy == 'transformer' else 0))
    epochs = int(opt.get('epochs', 20))
    if epochs < 1:
        raise ValueError('optimization.epochs must be positive')
    opt['total_steps'] = epochs * len(train_loader)
    step, start_epoch, best_score, stale, history = 0, 0, -float('inf'), 0, []
    if resume:
        checkpoint = load_checkpoint(resume, raw, optimizer, scaler)
        step, start_epoch = int(checkpoint['step']), int(checkpoint['epoch'])
        best_score, stale, history = checkpoint['best_score'], checkpoint.get('stale_epochs', 0), checkpoint.get('history', [])
        if checkpoint.get('world_size', world) != world:
            raise ValueError('full resume requires unchanged world size')
        if checkpoint.get('rng_states'):
            _restore_rng(checkpoint['rng_states'][rank])
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        train_dataset.epoch = epoch
        model.train()
        running, count = 0., 0
        train_paths, train_truth, train_logits = [], [], []
        progress = tqdm(train_loader, disable=rank != 0,
                        desc=f'train {epoch + 1}/{epochs}', ncols=progress_ncols())
        for batch_index, batch in enumerate(progress):
            _record_skipped(folder / f'logs/skipped_pairs_rank{rank}.jsonl', batch, epoch, batch_index)
            batch, _ = synchronize_training_batch(batch, device)
            if batch is None:
                continue
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _schedule(optimizer, step, opt)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mixed):
                output = model(batch)
                loss = _loss(output, batch['label'], criterion, task, branch_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite downstream loss')
            loss.backward()
            norms = {key: float(_grad_norm(module.parameters())) for key, module in
                [('video_grad_norm', raw.va.video_encoder), ('audio_grad_norm', raw.va.audio_encoder), ('fusion_grad_norm', raw.va.fusion_adapters)]}
            torch.nn.utils.clip_grad_norm_(raw.parameters(), float(opt.get('clip_grad_norm', 1)))
            optimizer.step()
            step += 1
            running += float(loss.detach()) * batch['label'].shape[0]
            count += batch['label'].shape[0]
            train_paths.extend(batch['video_path'])
            train_truth.extend(batch['label'].detach().cpu().tolist())
            train_logits.extend(output['logits'].detach().float().cpu().tolist())
            if rank == 0:
                progress.set_postfix(loss=f'{running / count:.4f}')
        if not count:
            raise RuntimeError(f'no usable VA training batches in epoch {epoch}')
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, (train_paths, train_truth, train_logits))
            train_paths = [item for paths, _, _ in gathered for item in paths]
            train_truth = [item for _, truths, _ in gathered for item in truths]
            train_logits = [item for _, _, logits in gathered for item in logits]
            totals = torch.tensor([running, count], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            running, count = totals.tolist()
        train_metrics, train_predictions, train_probabilities, _ = task_metrics(
            train_truth, train_logits, task, classes)
        metrics, records, predictions, probabilities, matrix, val_loss = evaluate(
            model, val_loader, device, task, classes, epoch + 1, epochs, criterion, branch_weight)
        score_key = cfg.get('best_metric', 'mae' if task == 'regression' else 'f1_macro')
        score_sign = -1 if cfg.get('best_metric_mode', 'min' if task == 'regression' else 'max') == 'min' else 1
        score = metrics[score_key] * score_sign
        improved = score > best_score
        if improved:
            best_score, stale = score, 0
        else:
            stale += 1
        row = dict(epoch=epoch + 1, step=step, train_loss=running / max(count, 1), val_loss=val_loss,
                   train_samples=len(train_truth), val_samples=len(records),
                   lr=float(optimizer.param_groups[0]['lr']),
                   weight_decay=float(optimizer.param_groups[0].get('weight_decay', 0)), **norms,
                   **{f'train_{key}': value for key, value in train_metrics.items() if not key.startswith('class_')},
                   **{f'val_{key}': value for key, value in metrics.items() if not key.startswith('class_')})
        if matrix is not None:
            row.update({f'cm_{i}_{j}': int(matrix[i, j]) for i in range(classes) for j in range(classes)})
        if raw.strategy == 'separate_logits':
            row.update({f'branch_weight_{mode}': float(raw.branch_weights[i]) for i, mode in enumerate(raw.va.enabled_modes)})
        history.append(row)
        if rank == 0:
            # The shared writer handles older checkpoint rows with fewer columns.
            write_history_csv(history, folder / 'logs/history.csv')
            with (folder / 'logs/predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['pair_id', 'video_path', 'audio_path', 'truth', 'prediction', 'probabilities', 'branch_logits'])
                writer.writeheader()
                for index, record in enumerate(records):
                    writer.writerow({**{k: record[k] for k in ('pair_id', 'video_path', 'audio_path', 'truth')},
                        'prediction': np.asarray(predictions[index]).tolist(), 'probabilities': np.asarray(probabilities[index]).tolist(),
                        'branch_logits': record['branches']})
            if matrix is not None:
                np.savetxt(folder / 'logs/confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
            if improved:
                _write_best_reports(folder / 'logs', epoch + 1, task, classes,
                                    train_paths, train_truth, np.asarray(train_logits),
                                    train_predictions, train_probabilities, records, metrics,
                                    predictions, probabilities, matrix, val_loss, raw.va.enabled_modes)
            save_metric_curves(history, folder / 'logs/metrics_curves.png')
            LOGGER.info(_epoch_summary(epoch + 1, task, row, best_score * score_sign))
        extra = dict(best_score=best_score, stale_epochs=stale, history=history, stage='finetune', task=task)
        _save_all(folder / 'latest.pt', model, optimizer, scaler, None, step, epoch + 1, args, 0, **extra)
        if improved:
            _save_all(folder / 'best.pt', model, optimizer, scaler, None, step, epoch + 1, args, 0, **extra)
        patience = int(cfg.get('early_stopping_patience', 0))
        if patience > 0 and stale >= patience:
            break
    if distributed:
        dist.barrier()
    return {'latest_checkpoint': folder / 'latest.pt', 'best_checkpoint': folder / 'best.pt'}
