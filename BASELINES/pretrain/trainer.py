"""The lifecycle is shared; each adapter owns its native self-supervised loss."""
import numpy as np
import json
import torch

from BASELINES.build_model import build_model
from BASELINES.common.data import make_loaders
from BASELINES.common.runtime import (setup, write_config, write_json, load_file,
                                      move, update, fit, parameter_counts, input_summary,
                                      check_evaluation_protocol, evaluation_rng)


def main(cfg, args):
    output, logger, device = setup(cfg)
    if not args.resume and not args.evaluate and (output / 'last.pt').exists():
        raise FileExistsError(f'{output}: use --resume or a new experiment name')
    model = build_model(cfg, pretrain=True, initialize=not bool(args.resume or args.evaluate)).to(device)
    loaders = make_loaders(cfg, model.input_spec, pretrain=True)
    cfg['parameter_counts'], cfg['native_input'] = parameter_counts(model), model.input_spec
    if not args.evaluate or not (output / 'config.yaml').exists():
        write_config(output, cfg)
    logger.info('model=%s route=%s input=%s parameters=%s', cfg['model']['id'], cfg['route'], model.input_spec, cfg['parameter_counts'])
    def epoch_fn(loader, optimizer, scheduler, scaler, step):
        losses, norms, samples = [], [], 0
        precision = cfg.get('precision', 'float32')
        for index, batch in enumerate(loader):
            if index >= int(cfg.get('max_batches', len(loader))):
                break
            prepared = [move(model.prepare_views(s['sentences'][0]) if hasattr(model, 'prepare_views')
                             else model.prepare(s['sentences'][0][0]), device) for s in batch]
            with torch.autocast(device.type, enabled=precision != 'float32',
                                dtype=torch.bfloat16 if precision == 'bfloat16' else torch.float16):
                loss = model.pretrain_loss(prepared, step+index)
            if not hasattr(model, '_io_check'):
                model._io_check = input_summary(batch[0]['sentences'][0][0], prepared[0], loss)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native pretraining loss')
            if optimizer is not None:
                norms.append(update(loss, model, optimizer, scheduler, scaler,
                                    clip=float(cfg.get('optimization', {}).get('clip_grad_norm', 1.))))
                if hasattr(model, 'after_update'):
                    model.after_update(step+index+1)
            losses.append(float(loss.detach()))
            samples += len(batch)
        if not losses:
            raise ValueError('Empty pretraining loader')
        return dict(loss=float(np.mean(losses)), samples=samples, updates=len(norms),
                    gradient_norm=float(np.mean(norms)) if norms else 0.)
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
        write_json(report_path, report)
        return metrics
    return fit(model, loaders, cfg, output, logger, device, args.resume, epoch_fn)
