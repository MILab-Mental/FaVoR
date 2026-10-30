"""The lifecycle is shared; each adapter owns its native self-supervised loss."""
import numpy as np
import json
from itertools import islice
import torch

from BASELINES.build_model import build_model
from BASELINES.common.data import make_loaders
from BASELINES.common.runtime import (setup, write_config, write_json, load_file,
                                      move, update, fit, parameter_counts, input_summary,
                                      check_evaluation_protocol, evaluation_rng)


def forward_batch(model,batch,step,device):
    prepared=[move(model.prepare_views(s['sentences'][0]) if hasattr(model,'prepare_views') else model.prepare(s['sentences'][0][0]),device) for s in batch]
    loss=model.pretrain_loss(prepared,step)
    if not hasattr(model,'_io_check'):
        model._io_check=input_summary(batch[0]['sentences'][0][0],prepared[0],loss)
    return loss


def main(cfg, args, model=None):
    output, logger, device = setup(cfg)
    if not args.resume and not args.evaluate and (output / 'latest.pt').exists():
        raise FileExistsError(f'{output}: use --resume or a new experiment name')
    model = (model if model is not None else build_model(cfg, pretrain=True, initialize=not bool(args.resume or args.evaluate))).to(device)
    loaders = make_loaders(cfg, model.input_spec, pretrain=True)
    cfg['parameter_counts'], cfg['native_input'] = parameter_counts(model), model.input_spec
    if not args.evaluate or not (output / 'config.yaml').exists():
        write_config(output, cfg)
    logger.info('model=%s route=%s input=%s parameters=%s', cfg['model']['id'], cfg['route'], model.input_spec, cfg['parameter_counts'])
    from BASELINES.common import distributed
    from contextlib import nullcontext
    training_step=distributed.training_step(model,lambda model,batch,step:forward_batch(model,batch,step,device)) if not args.evaluate else None
    def epoch_fn(loader,optimizer,scheduler,scaler,step):
        losses,norms=[],[]
        precision=cfg.get('precision','float32')
        accumulation=int(cfg['optimization'].get('gradient_accumulation',1))
        batches=min(len(loader),int(cfg.get('max_batches',len(loader))))
        for index,batch in enumerate(islice(loader,batches)):
            end=(index+1)%accumulation==0 or index+1==batches
            group=min(accumulation,batches-index//accumulation*accumulation)
            sync=training_step.no_sync() if optimizer is not None and not end and hasattr(training_step,'no_sync') else nullcontext()
            with sync:
                with torch.autocast(device.type,enabled=precision!='float32',dtype=torch.bfloat16 if precision=='bfloat16' else torch.float16):
                    loss=training_step(batch,step+index//accumulation) if optimizer is not None else forward_batch(model,batch,step+index//accumulation,device)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native pretraining loss')
                if optimizer is not None:
                    norm=update(loss,model,optimizer,scheduler,scaler,clip=float(cfg['optimization'].get('clip_grad_norm',1.)),accumulation=group,perform_step=end)
                    if norm is not None:
                        norms.append(norm)
                        if hasattr(model,'after_update'):model.after_update(step+len(norms))
            losses.append((float(loss.detach()),len(batch)))
        parts=distributed.gather((losses,norms))
        all_losses=[v for p in parts for v in p[0]]
        all_norms=[v for p in parts for v in p[1]]
        if not all_losses:raise ValueError('Empty pretraining loader')
        samples=sum(n for _,n in all_losses)
        return dict(loss=sum(v*n for v,n in all_losses)/samples,samples=samples,updates=len(norms),gradient_norm=float(np.mean(all_norms)) if all_norms else 0.)
    if args.evaluate:
        if args.evaluate not in loaders or not args.checkpoint:
            raise ValueError(f'No {args.evaluate} split in the pretraining manifest; available: {list(loaders)}')
        checkpoint = load_file(args.checkpoint)
        check_evaluation_protocol(checkpoint, cfg)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        with evaluation_rng(loaders[args.evaluate]), torch.no_grad():
            metrics = epoch_fn(loaders[args.evaluate], None, None, None, checkpoint['step'])
        report_path = output / 'metrics.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.setdefault('evaluations', {})[args.evaluate] = dict(checkpoint=args.checkpoint, metrics=metrics,
            io_check=getattr(model, '_io_check', {}), evaluation_seed=1701,
            scope='smoke_subset' if cfg['data'].get('limit') or cfg.get('max_batches') else 'full')
        if not cfg.get('no_save') and distributed.rank()==0:write_json(report_path, report)
        return metrics
    return fit(model, loaders, cfg, output, logger, device, args.resume, epoch_fn)
