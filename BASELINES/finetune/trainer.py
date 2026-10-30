"""Supervised training and instance-level validation/test using FAVOR metrics."""
import json
from itertools import islice

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


def forward_batch(model, batch, criterion, cfg, device):
    task = cfg['data']['task']
    labels = torch.as_tensor(np.stack([s['label'] for s in batch]),
        dtype=torch.long if task=='classification' else torch.float32, device=device)
    predicted = model.predict_samples(batch,task,device) if hasattr(model,'predict_samples') else [predict_sample(model,s,task,device) for s in batch]
    logits = torch.cat(predicted)
    loss = criterion(logits.squeeze(-1) if task=='regression' else logits,labels)
    return loss,logits,labels


def run_epoch(model, loader, criterion, cfg, device, optimizer=None, scheduler=None, scaler=None,
              step=0, predictions_path=None, training_step=None):
    from contextlib import nullcontext
    from BASELINES.common import distributed
    task = cfg['data']['task']
    truth, outputs, paths, losses, norms = [],[],[],[],[]
    precision = cfg.get('precision','float32')
    accumulation = int(cfg.get('optimization',{}).get('gradient_accumulation',1))
    batches = min(len(loader),int(cfg.get('max_batches',len(loader))))
    for index,batch in enumerate(islice(loader,batches)):
        end = (index+1)%accumulation==0 or index+1==batches
        group = min(accumulation,batches-index//accumulation*accumulation)
        sync = training_step.no_sync() if optimizer is not None and not end and hasattr(training_step,'no_sync') else nullcontext()
        with sync:
            with torch.autocast(device.type,enabled=precision!='float32',dtype=torch.bfloat16 if precision=='bfloat16' else torch.float16):
                loss,logits,labels = training_step(batch,step+index//accumulation) if optimizer is not None and training_step else forward_batch(model,batch,criterion,cfg,device)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite logits/loss')
            if optimizer is not None:
                norm=update(loss,model,optimizer,scheduler,scaler,
                    clip=float(cfg.get('optimization',{}).get('clip_grad_norm',1.)),accumulation=group,perform_step=end)
                if norm is not None:norms.append(norm)
        losses.extend([float(loss.detach())]*len(batch))
        truth.extend(labels.cpu().numpy())
        outputs.extend(logits.detach().float().cpu().numpy())
        paths.extend(s['id'] for s in batch)
    parts=distributed.gather((truth,outputs,paths,losses,norms))
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
    if predictions_path and distributed.rank()==0:
        if task=='classification':write_predictions_csv(predictions_path,paths,truth,predictions,logits,probabilities)
        elif task=='multi_label_classification':write_multilabel_predictions_csv(predictions_path,paths,truth,predictions,logits,probabilities)
        else:write_regression_predictions_csv(predictions_path,paths,truth,predictions)
    expected=min(len(loader.dataset),int(cfg['max_batches'])*loader.batch_size*distributed.world_size()) if cfg.get('max_batches') else len(loader.dataset)
    if optimizer is None and len(paths)!=expected:raise AssertionError(f'{len(paths)} predictions != {expected} valid instances')
    return metrics


def main(cfg, args, model=None):
    output, logger, device = setup(cfg)
    if not args.resume and not args.evaluate and (output / 'best.pt').exists():
        raise FileExistsError(f'{output} already has a run; use --resume or a new experiment name')
    model = (model if model is not None else build_model(cfg, initialize=not bool(args.resume or args.evaluate))).to(device)
    if not args.evaluate:
        from BASELINES.common.distributed import checkpoint_blocks
        checkpoint_blocks(model)
    if cfg['data'].get('num_sentence') and not cfg.get('no_save'):
        cfg['data']['batch_size']=min(2,int(cfg['data'].get('batch_size',8)))
        cfg['optimization']['gradient_accumulation']=8//cfg['data']['batch_size']
    loaders = make_loaders(cfg, model.input_spec)
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
        if not cfg.get('no_save') and distributed.rank()==0:write_json(report_path, report)
        logger.info('%s samples=%d loss=%.6g', args.evaluate, metrics['samples'], metrics['loss'])
        return metrics
    return fit(model, loaders, cfg, output, logger, device, args.resume, epoch_fn)
