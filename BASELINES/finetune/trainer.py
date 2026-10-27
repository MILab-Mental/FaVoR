"""Supervised training and instance-level validation/test using FAVOR metrics."""
import json

import numpy as np
import torch

from app.finetune_a.train import _criterion, _regression_metrics
from utils.classification_metrics import (classification_metrics, multilabel_metrics,
    write_predictions_csv, write_multilabel_predictions_csv, write_regression_predictions_csv)
from BASELINES.build_model import build_model
from BASELINES.common.data import make_loaders
from BASELINES.common.runtime import (setup, write_config, write_json, load_file,
                                      update, fit, parameter_counts,
                                      check_evaluation_protocol, evaluation_rng)


def predict_sample(model, sample, task, device):
    return model.predict_sample(sample, task, device)


def metrics_for(task, truth, logits, classes):
    if task == 'classification':
        metrics, _, predictions, probabilities = classification_metrics(truth, logits, classes)
    elif task == 'multi_label_classification':
        metrics, predictions, probabilities = multilabel_metrics(truth, logits, classes)
    else:
        predictions, probabilities = logits[:, 0], None
        metrics = _regression_metrics(predictions, truth)
    return metrics, predictions, probabilities


def run_epoch(model, loader, criterion, cfg, device, optimizer=None, scheduler=None, scaler=None, step=0, predictions_path=None):
    task = cfg['data']['task']
    truth, outputs, paths, losses, norms = [], [], [], [], []
    precision = cfg.get('precision', 'float32')
    amp_dtype = torch.bfloat16 if precision == 'bfloat16' else torch.float16
    for index, batch in enumerate(loader):
        if index >= int(cfg.get('max_batches', len(loader))):
            break
        labels = torch.as_tensor(np.stack([s['label'] for s in batch]),
            dtype=torch.long if task == 'classification' else torch.float32, device=device)
        with torch.autocast(device.type, enabled=precision != 'float32', dtype=amp_dtype):
            predicted = [predict_sample(model, sample, task, device) for sample in batch]
            if isinstance(predicted[0], dict):
                output = dict(logits=torch.cat([p['logits'] for p in predicted]),
                    branches={k:torch.cat([p['branches'][k] for p in predicted]) for k in predicted[0]['branches']})
                logits = output['logits']
                loss = model.supervised_loss(output, labels, criterion, task)
            else:
                logits = torch.cat(predicted)
                loss = criterion(logits.squeeze(-1) if task == 'regression' else logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite logits/loss')
        if optimizer is not None:
            norms.append(update(loss, model, optimizer, scheduler, scaler,
                                clip=float(cfg.get('optimization', {}).get('clip_grad_norm', 1.))))
        losses.extend([float(loss.detach())] * len(batch))
        truth.extend(labels.cpu().numpy())
        outputs.extend(logits.detach().float().cpu().numpy())
        paths.extend(s['id'] for s in batch)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Empty evaluation or duplicate instance predictions')
    truth, logits = np.asarray(truth), np.asarray(outputs)
    metrics, predictions, probabilities = metrics_for(task, truth, logits, int(cfg['data']['num_class']))
    metrics.update(loss=float(np.mean(losses)), samples=len(paths), updates=len(norms))
    if norms:
        metrics['gradient_norm'] = float(np.mean(norms))
    if predictions_path:
        if task == 'classification':
            write_predictions_csv(predictions_path, paths, truth, predictions, logits, probabilities)
        elif task == 'multi_label_classification':
            write_multilabel_predictions_csv(predictions_path, paths, truth, predictions, logits, probabilities)
        else:
            write_regression_predictions_csv(predictions_path, paths, truth, predictions)
    expected = min(len(loader.dataset), int(cfg.get('max_batches', len(loader))) * loader.batch_size)
    if len(paths) != expected:
        raise AssertionError(f'{len(paths)} predictions != {expected} valid instances')
    return metrics


def main(cfg, args):
    output, logger, device = setup(cfg)
    if not args.resume and not args.evaluate and (output / 'last.pt').exists():
        raise FileExistsError(f'{output} already has a run; use --resume or a new experiment name')
    model = build_model(cfg, initialize=not bool(args.resume or args.evaluate)).to(device)
    loaders = make_loaders(cfg, model.input_spec)
    criterion = _criterion(cfg['data']['task'], cfg.get('optimization', {}), loaders['train'].dataset,
                           int(cfg['data']['num_class']), device)
    cfg['parameter_counts'] = parameter_counts(model)
    cfg['native_input'] = model.input_spec
    if not args.evaluate or not (output / 'config.yaml').exists():
        write_config(output, cfg)
    logger.info('model=%s device=%s native_input=%s parameters=%s', cfg['model']['id'], device, model.input_spec, cfg['parameter_counts'])
    def epoch_fn(loader, optimizer, scheduler, scaler, step):
        return run_epoch(model, loader, criterion, cfg, device, optimizer, scheduler, scaler, step)
    if args.evaluate:
        if args.evaluate not in loaders:
            raise ValueError(f'The task CSV has no {args.evaluate} split; available: {list(loaders)}')
        if not args.checkpoint:
            raise ValueError('--evaluate requires --checkpoint best.pt')
        checkpoint = load_file(args.checkpoint)
        check_evaluation_protocol(checkpoint, cfg)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        with evaluation_rng(loaders[args.evaluate]), torch.no_grad():
            metrics = run_epoch(model, loaders[args.evaluate], criterion, cfg, device,
                predictions_path=output / 'predictions.csv' if cfg.get('save_predictions', False) else None)
        report_path = output / 'metrics.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.setdefault('evaluations', {})[args.evaluate] = dict(checkpoint=args.checkpoint, metrics=metrics,
            io_check=getattr(model, '_io_check', {}),
            scope='smoke_subset' if cfg['data'].get('limit') or cfg.get('max_batches') else 'full')
        write_json(report_path, report)
        logger.info('%s samples=%d loss=%.6g', args.evaluate, metrics['samples'], metrics['loss'])
        return metrics
    return fit(model, loaders, cfg, output, logger, device, args.resume, epoch_fn)
