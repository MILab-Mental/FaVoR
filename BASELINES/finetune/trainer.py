"""Supervised training and instance-level validation/test using FAVOR metrics."""
import csv
import json
import os
from itertools import islice
from time import perf_counter

import numpy as np
import torch

from app.finetune_a.train import _criterion, _regression_metrics
from utils.classification_metrics import (classification_metrics, multilabel_metrics,
    write_predictions_csv, write_multilabel_predictions_csv, write_regression_predictions_csv)
from BASELINES.build_model import build_model
from BASELINES.common.data import make_loaders
from BASELINES.common.runtime import (setup, write_config, write_json, load_file,
                                      update, fit, parameter_counts,
                                      check_evaluation_protocol, evaluation_rng,
                                      uses_mental_test_selection, NonfiniteEvaluationError)


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


def forward_batch(model, batch, criterion, cfg, device):
    task = cfg['data']['task']
    labels = torch.as_tensor(np.stack([s['label'] for s in batch]),
        dtype=torch.long if task=='classification' else torch.float32, device=device)
    predicted = model.predict_samples(batch,task,device) if hasattr(model,'predict_samples') else [predict_sample(model,s,task,device) for s in batch]
    logits = torch.cat(predicted)
    loss = criterion(logits.squeeze(-1) if task=='regression' else logits,labels)
    return loss,logits,labels


def write_inference_epoch(output, epoch, write_predictions, args):
    """Replace one epoch atomically so retrying an epoch cannot duplicate rows."""
    rows_file = output.with_suffix('.epoch.tmp')
    merged_file = output.with_suffix('.csv.tmp')
    try:
        write_predictions(rows_file, *args)
        with rows_file.open(newline='', encoding='utf-8') as current, merged_file.open(
                'w', newline='', encoding='utf-8') as merged:
            reader = csv.DictReader(current)
            fields = ['epoch', *reader.fieldnames]
            writer = csv.DictWriter(merged, fieldnames=fields)
            writer.writeheader()
            if output.exists():
                with output.open(newline='', encoding='utf-8') as previous:
                    existing = csv.DictReader(previous)
                    if existing.fieldnames != fields:
                        raise ValueError(f'Inference CSV columns changed: {output}')
                    writer.writerows(row for row in existing if int(row['epoch']) != epoch)
            writer.writerows(dict(row, epoch=epoch) for row in reader)
        merged_file.replace(output)
    finally:
        rows_file.unlink(missing_ok=True)
        merged_file.unlink(missing_ok=True)


def run_epoch(model, loader, criterion, cfg, device, optimizer=None, scheduler=None, scaler=None,
              step=0, predictions_path=None, training_step=None, prediction_epoch=None):
    from contextlib import nullcontext
    from BASELINES.common import distributed
    task = cfg['data']['task']
    truth, outputs, paths, losses, norms = [],[],[],[],[]
    evaluation_error = None
    precision = cfg.get('precision','float32')
    accumulation = int(cfg.get('optimization',{}).get('gradient_accumulation',1))
    batches = min(len(loader),int(cfg.get('max_batches',len(loader))))
    profile = optimizer is not None and os.environ.get('BASELINES_TIMING') == '1'
    if profile:
        model._profile_prepare_s = 0.0
        data_wait_s = batch_s = first_batch_s = 0.0
        next_requested = perf_counter()
    for index,batch in enumerate(islice(loader,batches)):
        if profile:
            batch_started = perf_counter()
            waited = batch_started - next_requested
            data_wait_s += waited
            if index == 0:
                first_batch_s = waited
        end = (index+1)%accumulation==0 or index+1==batches
        group = min(accumulation,batches-index//accumulation*accumulation)
        sync = training_step.no_sync() if optimizer is not None and not end and hasattr(training_step,'no_sync') else nullcontext()
        with sync:
            with torch.autocast(device.type,enabled=precision!='float32',dtype=torch.bfloat16 if precision=='bfloat16' else torch.float16):
                loss,logits,labels = training_step(batch,step+index//accumulation) if optimizer is not None and training_step else forward_batch(model,batch,criterion,cfg,device)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                finite = False
                if optimizer is None and precision != 'float32':
                    # A rare reduced-precision overflow must not discard a finished
                    # run. Recompute only the affected evaluation batch in fp32.
                    with torch.autocast(device.type, enabled=False):
                        loss, logits, labels = forward_batch(model, batch, criterion, cfg, device)
                    finite = bool(torch.isfinite(logits).all() and torch.isfinite(loss))
                    print('EVAL_FP32_FALLBACK finite=', finite, 'samples=', len(batch), flush=True)
                if not finite:
                    print('NONFINITE batch samples=', [sample['id'] for sample in batch],
                          'loss=', loss.item(), 'logits=', logits.flatten()[:8].tolist(), flush=True)
                    if optimizer is not None:
                        raise FloatingPointError('Nonfinite logits/loss')
                    evaluation_error = f'Nonfinite evaluation logits/loss on rank {distributed.rank()}, batch {index + 1}'
                    break
            if optimizer is not None:
                norm=update(loss,model,optimizer,scheduler,scaler,
                    clip=float(cfg.get('optimization',{}).get('clip_grad_norm',1.)),accumulation=group,perform_step=end)
                if norm is not None:norms.append(norm)
        losses.extend([float(loss.detach())]*len(batch))
        truth.extend(labels.cpu().numpy())
        outputs.extend(logits.detach().float().cpu().numpy())
        paths.extend(s['id'] for s in batch)
        if profile:
            batch_s += perf_counter() - batch_started
            if index + 1 in (64, batches):
                prepare_s = model._profile_prepare_s
                print(f'TIMING train rank={distributed.rank()} batches={index + 1}/{batches} '
                      f'first_batch_s={first_batch_s:.2f} data_wait_s={data_wait_s:.2f} '
                      f'prepare_s={prepare_s:.2f} compute_sync_s={batch_s - prepare_s:.2f}',
                      flush=True)
            next_requested = perf_counter()
    if profile:
        del model._profile_prepare_s
    # Evaluation has no per-batch collectives. Share errors at the same
    # collective as successful predictions so both ranks report the failure.
    parts=distributed.gather((truth,outputs,paths,losses,norms,evaluation_error))
    if error := next((part[5] for part in parts if part[5]), None):
        raise NonfiniteEvaluationError(error)
    truth,outputs,paths,losses,all_norms=([v for part in parts for v in part[i]] for i in range(5))
    if not paths:raise ValueError('Empty evaluation')
    # DistributedSampler may repeat one training row to balance ranks.
    unique={name:i for i,name in enumerate(paths)}
    if optimizer is None and len(unique)!=len(paths):raise ValueError('Duplicate evaluation predictions')
    indices=list(unique.values())
    truth,logits=np.asarray(truth)[indices],np.asarray(outputs)[indices]
    paths=[paths[i] for i in indices]
    metrics,predictions,probabilities=metrics_for(task,truth,logits,int(cfg['data']['num_class']))
    metrics.update(loss=float(np.mean(losses)),samples=len(paths),updates=len(norms))
    if all_norms:metrics['gradient_norm']=float(np.mean(all_norms))
    expected=min(len(loader.dataset),int(cfg['max_batches'])*loader.batch_size*distributed.world_size()) if cfg.get('max_batches') else len(loader.dataset)
    if optimizer is None and len(paths)!=expected:raise AssertionError(f'{len(paths)} predictions != {expected} valid instances')
    if predictions_path and distributed.rank()==0:
        args = (paths, truth, predictions)
        if task == 'regression':
            writer = write_regression_predictions_csv
        else:
            writer = write_predictions_csv if task == 'classification' else write_multilabel_predictions_csv
            args += (logits, probabilities)
        if prediction_epoch is None:
            writer(predictions_path, *args)
        else:
            write_inference_epoch(predictions_path, prediction_epoch, writer, args)
    return metrics


def main(cfg, args, model=None):
    profile = os.environ.get('BASELINES_TIMING') == '1'
    started = perf_counter() if profile else None
    output, logger, device = setup(cfg)
    if not args.resume and not args.evaluate and ((output / 'best.pt').exists() or
            (uses_mental_test_selection(cfg) and (output / 'latest.pt').exists())):
        raise FileExistsError(f'{output} already has a run; use --resume or a new experiment name')
    model = (model if model is not None else build_model(cfg, initialize=not bool(args.resume or args.evaluate))).to(device)
    if not args.evaluate:
        from BASELINES.common.distributed import checkpoint_blocks
        checkpoint_blocks(model)
    if cfg['data'].get('num_sentence') and not cfg.get('no_save'):
        cfg['data']['batch_size']=min(2,int(cfg['data'].get('batch_size',8)))
        cfg['optimization']['gradient_accumulation']=8//cfg['data']['batch_size']
    if profile:
        loader_started = perf_counter()
    loaders = make_loaders(cfg, model.input_spec)
    if profile:
        logger.info('TIMING setup_s=%.2f loader_s=%.2f phase=%s',
                    loader_started - started, perf_counter() - loader_started,
                    args.evaluate or 'train')
    criterion = _criterion(cfg['data']['task'], cfg.get('optimization', {}), loaders['train'].dataset,
                           int(cfg['data']['num_class']), device)
    cfg['parameter_counts'] = parameter_counts(model)
    cfg['native_input'] = model.input_spec
    if not args.evaluate or not (output / 'config.yaml').exists():
        write_config(output, cfg)
    logger.info('model=%s device=%s native_input=%s parameters=%s', cfg['model']['id'], device, model.input_spec, cfg['parameter_counts'])
    from BASELINES.common import distributed
    training_step=distributed.training_step(model,lambda model,batch,step:forward_batch(model,batch,criterion,cfg,device)) if not args.evaluate else None
    def epoch_fn(loader, optimizer, scheduler, scaler, step):
        return run_epoch(model, loader, criterion, cfg, device, optimizer, scheduler, scaler, step,training_step=training_step)
    def test_epoch_fn(loader, step, epoch):
        predictions_path = None if cfg.get('no_save') else output / 'infer.csv'
        return run_epoch(model, loader, criterion, cfg, device, step=step,
                         predictions_path=predictions_path, prediction_epoch=epoch)
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
                predictions_path=output / 'predictions.csv' if cfg.get('save_predictions', False) and not cfg.get('no_save') else None)
        report_path = output / 'metrics.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.setdefault('evaluations', {})[args.evaluate] = dict(checkpoint=args.checkpoint, metrics=metrics,
            io_check=getattr(model, '_io_check', {}),
            scope='smoke_subset' if cfg['data'].get('limit') or cfg.get('max_batches') else 'full')
        if (checkpoint.get('best') or {}).get('selection') == 'test':
            report['evaluations'][args.evaluate]['selection_split'] = 'test'
        if not cfg.get('no_save') and distributed.rank()==0:write_json(report_path, report)
        logger.info('%s samples=%d loss=%.6g', args.evaluate, metrics['samples'], metrics['loss'])
        return metrics
    return fit(model, loaders, cfg, output, logger, device, args.resume, epoch_fn, test_epoch_fn=test_epoch_fn)
