"""Paired VA downstream training; shared metrics and task-label conventions."""
import csv
import logging
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from app.pretrain_audio_lejepa.train import _state, _unwrap, _schedule, _append_csv
from app.pretrain_va_lejepa.train import _optimizer, _save_all, _restore_rng, _grad_norm
from app.finetune_a.train import _criterion, _regression_metrics
from datasets.va_lejepa.finetune_dataset import AVCSVDataset
from datasets.va_lejepa import collate_va_lejepa, to_device
from models.va_lejepa import build_va_lejepa, load_checkpoint
from models.va_lejepa.finetune import VAFinetuner
from utils.classification_metrics import classification_metrics, multilabel_metrics, save_metric_curves

LOGGER = logging.getLogger(__name__)


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
def evaluate(model, loader, device, task, classes):
    model.eval()
    raw = _unwrap(model)
    records = []
    for batch in loader:
        output = raw(to_device(batch, device))
        for i, pair_id in enumerate(batch['pair_id']):
            records.append(dict(pair_id=pair_id, video_path=batch['video_path'][i], audio_path=batch['audio_path'][i],
                truth=batch['label'][i].cpu().tolist(), logits=output['logits'][i].float().cpu().tolist(),
                branches={mode: value[i].float().cpu().tolist() for mode, value in output['branches'].items()}))
    enabled, rank, world = _state()
    if enabled:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [r for shard in gathered for r in shard]
    if not records:
        raise ValueError('empty validation split')
    metrics, predictions, probabilities, matrix = task_metrics([r['truth'] for r in records], [r['logits'] for r in records], task, classes)
    for mode in raw.va.enabled_modes:
        branch_metrics, _, _, _ = task_metrics([r['truth'] for r in records], [r['branches'][mode] for r in records], task, classes)
        metrics.update({f'{mode}_{key}': value for key, value in branch_metrics.items()})
    return metrics, records, predictions, probabilities, matrix


def main(args):
    distributed, rank, world = _state()
    data, cfg, opt = dict(args['data']), args.get('finetune', {}), dict(args['optimization'])
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
    resume = meta.get('read_checkpoint') or (folder / 'latest.pt' if meta.get('auto_resume', True) and (folder / 'latest.pt').exists() else None)
    va = build_va_lejepa(args['model'], data, initialize=False)
    if not resume:
        path = cfg.get('pretrained_checkpoint')
        if not path:
            raise ValueError('finetune.pretrained_checkpoint is required')
        load_checkpoint(path, va, allow_new_branches=True)
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
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True)
    def loader(dataset, sample):
        return DataLoader(dataset, sampler=sample, batch_size=int(data.get('batch_size', 1)),
            num_workers=int(data.get('num_workers', 4)), collate_fn=collate_va_lejepa, pin_memory=True)
    train_loader = loader(train_dataset, sampler)
    val_loader = loader(val_dataset, EvaluationSampler(val_dataset, rank, world))
    if not len(train_loader):
        raise ValueError('empty train loader')
    # Same weighting/loss options as FAVOR audio finetuning.
    criterion = _criterion(task, opt, train_dataset, classes, device)
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
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _schedule(optimizer, step, opt)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mixed):
                output = model(batch)
                branch_weight = float(cfg.get('branch_loss_weight', .1 if raw.strategy == 'transformer' else 0))
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
        metrics, records, predictions, probabilities, matrix = evaluate(model, val_loader, device, task, classes)
        score_key = cfg.get('best_metric', 'mae' if task == 'regression' else 'f1_macro')
        score = metrics[score_key] * (-1 if cfg.get('best_metric_mode', 'min' if task == 'regression' else 'max') == 'min' else 1)
        improved = score > best_score
        if improved:
            best_score, stale = score, 0
        else:
            stale += 1
        row = dict(epoch=epoch + 1, step=step, train_loss=running / max(count, 1), **norms,
                   **{f'val_{key}': value for key, value in metrics.items()})
        if raw.strategy == 'separate_logits':
            row.update({f'branch_weight_{mode}': float(raw.branch_weights[i]) for i, mode in enumerate(raw.va.enabled_modes)})
        history.append(row)
        if rank == 0:
            # Rewrite from checkpoint history on resume; no duplicate/stale epochs.
            path = folder / 'logs/history.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            with (folder / 'logs/predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['pair_id', 'video_path', 'audio_path', 'truth', 'prediction', 'probabilities', 'branch_logits'])
                writer.writeheader()
                for index, record in enumerate(records):
                    writer.writerow({**{k: record[k] for k in ('pair_id', 'video_path', 'audio_path', 'truth')},
                        'prediction': np.asarray(predictions[index]).tolist(), 'probabilities': np.asarray(probabilities[index]).tolist(),
                        'branch_logits': record['branches']})
            if matrix is not None:
                np.savetxt(folder / 'logs/confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
            save_metric_curves(history, folder / 'logs/metrics_curves.png')
            LOGGER.info('VA finetune epoch=%d score=%s best=%s', epoch + 1, score, best_score)
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
