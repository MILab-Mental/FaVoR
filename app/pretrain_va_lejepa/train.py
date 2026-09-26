"""Step-based VA pretraining with shared encoders, DDP, AMP and full resume."""
import logging
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from app.pretrain_audio_lejepa.train import _state, _unwrap, _schedule, _reduce, _append_csv, _trim_history
from datasets.va_lejepa import make_va_lejepa_loader, to_device, synchronize_training_batch
from models.va_lejepa import build_va_lejepa, VABranchLoss, save_checkpoint, load_checkpoint
from models.video_lejepa import ModelEMA, embedding_statistics

LOGGER = logging.getLogger(__name__)


def _record_skipped(path, batch, epoch, batch_index):
    errors = batch.get('skipped_samples', [])
    if errors:
        with Path(path).open('a', encoding='utf-8') as handle:
            for error in errors:
                handle.write(json.dumps(dict(epoch=epoch, batch_index=batch_index, **error), ensure_ascii=False) + '\n')
    return len(errors)


def _optimizer(model, cfg):
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Also handles finetune wrapper's va. prefix.
        normalized = name.removeprefix('va.')
        family = 'video' if normalized.startswith('video_encoder.') else 'audio' if normalized.startswith('audio_encoder.') else 'head' if not name.startswith('va.') and hasattr(model, 'va') else 'new'
        no_decay = parameter.ndim <= 1 or 'norm' in name.lower() or name.endswith('bias')
        buckets.setdefault((family, no_decay), []).append(parameter)
    lr_keys = {'video': 'lr_video_pretrained', 'audio': 'lr_audio_pretrained', 'new': 'lr_new', 'head': 'lr_head'}
    groups = []
    for (family, no_decay), parameters in buckets.items():
        lr = float(cfg.get(lr_keys[family], cfg.get('lr_new', 1.5e-4)))
        groups.append(dict(params=parameters, lr=lr, base_lr=lr, weight_decay=0 if no_decay else float(cfg.get('weight_decay', .04)),
                           group_name=family + ('_no_decay' if no_decay else '_decay')))
    return torch.optim.AdamW(groups, betas=tuple(cfg.get('betas', (.9, .98))), eps=float(cfg.get('eps', 1e-8)))


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _save_all(path, model, optimizer, scaler, ema, step, epoch, config, batch_offset, **extra):
    enabled, rank, world = _state()
    states = [None] * world
    local = _rng_state()
    if enabled:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        save_checkpoint(path, model, optimizer, scaler, ema, step, epoch, config, rng_states=states,
                        batch_offset=batch_offset, world_size=world, **extra)


def _grad_norm(parameters):
    grads = [p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]
    return torch.stack(grads).sum().sqrt() if grads else torch.tensor(0.)


def _write_curves(path, output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in rows[0]:
        if key == 'loss' or key.endswith('_loss'):
            ax.plot([int(r['step']) for r in rows], [float(r[key]) for r in rows], label=key)
    ax.set(xlabel='optimizer step', ylabel='loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(args):
    distributed, rank, world = _state()
    meta, opt = args.get('meta', {}), args['optimization']
    total, accumulation = int(opt['total_steps']), int(opt.get('grad_accum_steps', 1))
    if total < 1 or accumulation < 1 or int(meta.get('save_every_steps', 5000)) < 0:
        raise ValueError('invalid step/accumulation/save configuration')
    seed = int(meta.get('seed', 0)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    folder = Path(args['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'logs').mkdir(exist_ok=True)
    history = folder / 'logs/history.csv'
    resume = meta.get('read_checkpoint') or (folder / 'latest.pt' if meta.get('auto_resume', True) and (folder / 'latest.pt').exists() else None)
    model = build_va_lejepa(args['model'], args['data'], initialize=not resume and (not distributed or rank == 0)).to(device)
    if distributed:
        # Broadcast before EMA initialization so every rank shadows identical weights.
        model = DDP(model, device_ids=[0] if device.type == 'cuda' else None)
    raw = _unwrap(model)
    optimizer = _optimizer(raw, opt)
    dtype = str(meta.get('dtype', 'bfloat16'))
    mixed = device.type == 'cuda' and dtype in {'bfloat16', 'float16'}
    amp_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=mixed and dtype == 'float16')
    ema_cfg = args.get('ema', {})
    ema = ModelEMA(raw, decay=float(ema_cfg.get('decay', .9999)), update_every=int(ema_cfg.get('update_every', 32)),
                   device=ema_cfg.get('device', 'cpu')) if ema_cfg.get('enabled', True) else None
    criterion = VABranchLoss(raw, args.get('loss', {})).to(device)
    step = epoch = offset = 0
    pending_rng = None
    if resume:
        checkpoint = load_checkpoint(resume, raw, optimizer, scaler, ema)
        if checkpoint.get('world_size', world) != world:
            raise ValueError('full resume requires unchanged world size')
        step, epoch, offset = int(checkpoint['step']), int(checkpoint['epoch']), int(checkpoint.get('batch_offset', 0))
        raw.initialization_report = checkpoint.get('initialization_report', {})
        pending_rng = checkpoint.get('rng_states', [None] * world)[rank]
        if rank == 0:
            _trim_history(history, step)
            _trim_history(folder / 'logs/data_skips.csv', step)
    if rank == 0 and history.exists() and history.stat().st_size:
        # Keep resumed histories readable after adding the timing column.
        with history.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields, rows = list(reader.fieldnames), list(reader)
        if 'step_seconds' not in fields:
            fields.insert(fields.index('duration_delta'), 'step_seconds')
            with history.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    data_cfg = dict(args['data'])
    data_cfg['seed'] = int(meta.get('seed', 0))
    data_cfg.setdefault('audio_augmentation', args.get('audio_augmentation', {}))
    _, loader, sampler = make_va_lejepa_loader(data_cfg, rank, world)
    if not len(loader):
        raise ValueError('empty VA loader')
    if offset > len(loader):
        raise ValueError('resume batch_offset is outside current loader')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    skipped_pairs = discarded_pairs = skipped_batches = 0
    skipped_report = folder / f'logs/skipped_pairs_rank{rank}.jsonl'
    started = time.perf_counter()
    while step < total:
        sampler.set_epoch(epoch)
        loader.dataset.epoch = epoch
        attempted_batches = processed_batches = 0
        for batch_index, batch in enumerate(loader):
            if batch_index < offset:
                continue
            if pending_rng is not None:
                _restore_rng(pending_rng)
                pending_rng = None
            attempted_batches += 1
            skipped_pairs += _record_skipped(skipped_report, batch, epoch, batch_index)
            batch, discarded = synchronize_training_batch(batch, device)
            discarded_pairs += discarded
            if batch is None:
                skipped_batches += 1
                continue
            processed_batches += 1
            batch = to_device(batch, device)
            should_step = (micro + 1) % accumulation == 0
            sync = model.no_sync() if distributed and not should_step else nullcontext()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            with sync:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=mixed):
                    output = model(batch)
                    losses = criterion(output)
                forward_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('nonfinite VA loss')
                scaler.scale(losses['loss'] / accumulation).backward()
            backward_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            micro += 1
            if not should_step:
                continue
            _schedule(optimizer, step, opt)
            scaler.unscale_(optimizer)
            norms = {family + '_grad_norm': _grad_norm(module.parameters()).to(device) for family, module in
                     [('video', raw.video_encoder), ('audio', raw.audio_encoder), ('fusion', raw.fusion_adapters)]}
            norms.update({mode + '_grad_norm': _grad_norm(raw.fusion_adapters[mode].parameters()).to(device) for mode in raw.enabled_modes})
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), float(opt.get('clip_grad_norm', 1)))
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < old_scale:
                continue
            step += 1
            if ema:
                ema.update(raw, step)
            log_every = max(1, int(meta.get('log_every_steps', 50)))
            if step == 1 or step % log_every == 0 or step == total:
                skip_summary = dict(step=step, epoch=epoch,
                    skipped_pairs=int(round(_reduce(torch.tensor(skipped_pairs, device=device)) * world)),
                    ddp_discarded_pairs=int(round(_reduce(torch.tensor(discarded_pairs, device=device)) * world)),
                    skipped_batches=skipped_batches)
                row = dict(step=step, epoch=epoch, loss=_reduce(losses['loss']), grad_norm=_reduce(norm),
                           **{key: _reduce(value) for key, value in norms.items()})
                for mode, values in losses['branches'].items():
                    row.update({f'{mode}_{key}': _reduce(value) for key, value in values.items()})
                    embeddings = torch.cat((output[mode]['global'], output[mode]['local']), 1)
                    row.update({f'{mode}_{key}': _reduce(value) for key, value in embedding_statistics(embeddings.detach()).items()})
                row.update({f'lr_{g["group_name"]}': g['lr'] for g in optimizer.param_groups})
                step_seconds = time.perf_counter() - started
                diagnostics = batch['sync_diagnostics']
                row.update(step_seconds=step_seconds, duration_delta=max(d['duration_delta'] for d in diagnostics),
                           start_delta=max(d['start_delta'] for d in diagnostics),
                           max_frame_sampling_drift_seconds=max(d.get('max_frame_sampling_drift_seconds', 0) for d in diagnostics),
                           max_audio_boundary_rounding_seconds=max(d.get('max_audio_boundary_rounding_seconds', 0) for d in diagnostics),
                           repeat_ratio=sum(d['repeated'] for d in diagnostics) / len(diagnostics),
                           pad_ratio=sum(d['padded'] for d in diagnostics) / len(diagnostics),
                           forward_peak_bytes=forward_peak, backward_peak_bytes=backward_peak,
                           optimizer_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                           samples_per_second=world * batch['global']['video'].shape[0] * accumulation / max(step_seconds, 1e-9))
                if rank == 0:
                    _append_csv(history, row)
                    _append_csv(folder / 'logs/data_skips.csv', skip_summary)
                    modes = tuple(losses['branches'])
                    def branch_mean(metric):
                        return sum(row[f'{mode}_{metric}'] for mode in modes) / len(modes)
                    LOGGER.info(
                        'loss=%.5f inv=%.5f sigreg=%.5f std=%.4f rank=%.2f lr=%.2e wd=%.3g grad=%.3f clips/s=%.2f',
                        row['loss'], branch_mean('invariance_loss'), branch_mean('sigreg_loss'),
                        branch_mean('embedding_std'), branch_mean('rankme'),
                        max(group['lr'] for group in optimizer.param_groups),
                        max(group['weight_decay'] for group in optimizer.param_groups),
                        row['grad_norm'], row['samples_per_second'])
            started = time.perf_counter()
            save_every = int(meta.get('save_every_steps', 5000))
            if (save_every and step % save_every == 0) or step == total:
                _save_all(folder / 'latest.pt', model, optimizer, scaler, ema, step, epoch, args, batch_index + 1)
                if rank == 0:
                    _write_curves(history, folder / 'logs/metrics_curves.png')
            if step >= total:
                break
        if attempted_batches and not processed_batches:
            raise RuntimeError(f'no usable VA training batches in epoch {epoch}; see logs/skipped_pairs_rank*.jsonl')
        epoch += 1
        offset = 0
    if distributed:
        dist.barrier()
    return {'latest_checkpoint': folder / 'latest.pt', 'step': step}
