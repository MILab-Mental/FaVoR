"""Small, shared run/config/checkpoint utilities. Run entry points with python -m."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import random
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault('USE_TF', '0')

from app.main import load_config, _coerce_override_value

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path('/home/data/sdd/mental/OUTPUTS/对比实验输出')
WEIGHTS_ROOT = Path('/home/data/sdd/mental/OUTPUTS/对比实验初始化权重')


def path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def fingerprint(filename):
    filename = path(filename).resolve()
    stat = filename.stat()
    return _fingerprint(str(filename), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=128)
def _fingerprint(filename, size, mtime, ctime):
    digest = hashlib.sha256()
    with path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def arguments(stage):
    parser = argparse.ArgumentParser(description=f'FaVoR baselines: {stage}')
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=YAML')
    parser.add_argument('--resume', help='Restore latest.pt (pretraining) or an unfinished best.pt (finetuning)')
    parser.add_argument('--evaluate', choices=['validation', 'test'])
    parser.add_argument('--checkpoint', help='latest.pt (pretraining) or best.pt (finetuning) for evaluation')
    args = parser.parse_args()
    cfg = load_config(path(args.config))
    apply_overrides(cfg,args.set)
    cfg['stage'] = stage
    return args, cfg


def apply_overrides(cfg,overrides):
    for override in overrides:
        key, value = override.split('=', 1)
        destination = cfg
        keys = key.split('.')
        for part in keys[:-1]:
            destination = destination.setdefault(part, {})
        destination[keys[-1]] = _coerce_override_value(value)
    return cfg


def setup(cfg):
    from BASELINES.common import distributed
    distributed_device = distributed.initialize()
    from importlib.metadata import version, PackageNotFoundError
    cfg['software'] = {}
    for package in ('torch', 'torchaudio', 'torchvision', 'timm', 'transformers', 'numpy','fairseq','peft',
                    'nvidia-nccl-cu12','nvidia-nccl-cu13','dlib','scikit-image'):
        try:
            cfg['software'][package] = version(package)
        except PackageNotFoundError:
            pass
    seed = int(cfg.get('seed', 42))
    random.seed(seed + distributed.rank())
    np.random.seed(seed + distributed.rank())
    torch.manual_seed(seed + distributed.rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + distributed.rank())
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    output = path(cfg.get('output', OUTPUT_ROOT / cfg['stage'] / cfg['model']['id'] / cfg.get('experiment', 'default')))
    if not cfg.get('no_save') and distributed.rank()==0:
        output.mkdir(parents=True, exist_ok=True)
    cfg['output'] = str(output)
    logger = logging.getLogger('baselines')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:handler.close()
    logger.handlers.clear()
    handlers = [logging.StreamHandler()] if distributed.rank()==0 else [logging.NullHandler()]
    if not cfg.get('no_save') and distributed.rank()==0:
        handlers.append(logging.FileHandler(output / 'train.log'))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    return output, logger, distributed_device or torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))


def write_config(output, cfg):
    from BASELINES.common import distributed
    if cfg.get('no_save') or distributed.rank()!=0:
        return
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


def write_json(filename, value):
    # Undefined small-cohort AUC/R2 are JSON null, never invalid JSON NaN.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not math.isfinite(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x
    Path(filename).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False))


def parameter_counts(model):
    def count(module):
        return {'total': sum(p.numel() for p in module.parameters()),
                'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
    result = {'whole_run_model': count(model)}
    if hasattr(model, 'encoder'):
        scope = 'backbone_with_pretraining_components' if hasattr(model, 'export_encoder') else 'retained_encoder'
        result[scope] = count(model.encoder)
    if hasattr(model, 'head'):
        result['new_task_head'] = count(model.head)
    if hasattr(model, 'counted_components'):
        registered={id(p) for p in model.parameters()}
        external={}
        for name, module in model.counted_components().items():
            result[name] = count(module)
            external.update({id(p):p for p in module.parameters() if id(p) not in registered})
        if external:
            result['whole_run_including_external_components']=dict(
                total=result['whole_run_model']['total']+sum(p.numel() for p in external.values()),
                trainable=result['whole_run_model']['trainable']+sum(p.numel() for p in external.values() if p.requires_grad))
    return result


def strict_load(model, state, *, discard=(), allow_missing=()):
    """Reject every unaccounted-for key/shape; discarded upstream components are explicit."""
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    removed = [k for k in state if k.startswith(tuple(discard))] if discard else []
    state = {k: v for k, v in state.items() if k not in removed}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = {k: [list(state[k].shape), list(expected[k].shape)]
                for k in set(state) & set(expected) if state[k].shape != expected[k].shape}
    report = dict(missing=missing, unexpected=unexpected, mismatched=mismatch, discarded=removed)
    log_report = dict(report, discarded_count=len(removed), discarded=removed[:8])
    logging.getLogger('baselines').info('weight loading: %s', log_report)
    if unexpected or mismatch or any(not k.startswith(tuple(allow_missing)) for k in missing):
        raise ValueError(f'Checkpoint is incompatible: {report}')
    model.load_state_dict(state, strict=not missing)
    return report


def load_file(filename):
    if isinstance(filename,dict):return filename
    filename = path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {filename}')
    return torch.load(filename, map_location='cpu', weights_only=False, mmap=True)


def finalize_finetune_checkpoint(filename):
    """Keep one best model after test; interrupted runs retain full best state."""
    from BASELINES.common import distributed
    if distributed.rank()!=0:return
    state=load_file(filename)
    if state.get('training_complete'):return
    state={k:v for k,v in state.items() if k in ('model','encoder','config','epoch','step','best')}
    state['training_complete']=True
    temporary=Path(str(filename)+'.tmp')
    torch.save(state,temporary)
    temporary.replace(filename)


def optimizer_and_scheduler(model, cfg, steps):
    opt = cfg.get('optimization', {})
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad and (p.ndim<=1)==no_decay],
                 weight_decay=0. if no_decay else float(opt.get('weight_decay',.05))) for no_decay in (False,True)]
    optimizer=torch.optim.AdamW(groups,lr=float(opt.get('lr',1e-4)),betas=tuple(opt.get('betas',[.9,.95])),
                               fused=bool(opt.get('fused',True)) and next(model.parameters()).is_cuda)
    warmup = int(opt.get('warmup_steps', 0))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, steps - warmup))
        final = float(opt.get('final_lr_scale', 0.))
        return final + (1-final) * .5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(filename, model, optimizer, scheduler, scaler, loader, cfg, epoch, step, best, history,
                    resumable=True, validation_loader=None):
    from BASELINES.common import distributed
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                 epoch=epoch, step=step, best=best, history=history, config=cfg,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(), cuda=([torch.cuda.get_rng_state()] if distributed.active() else torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
                          cuda_local_only=distributed.active()),
                 loader_rng=loader.generator.get_state())
    if validation_loader is not None:
        state['validation_loader_rng'] = validation_loader.generator.get_state()
    state['rank_states'] = distributed.gather({k:state[k] for k in ('rng','loader_rng','validation_loader_rng') if k in state})
    if hasattr(model, 'export_encoder'):
        state['encoder'] = model.export_encoder()
    if not resumable:
        state = {k: v for k, v in state.items() if k in ('model', 'encoder', 'config', 'epoch', 'step', 'best')}
    if filename is None:
        return distributed.cpu_copy(state)
    if cfg.get('no_save'):
        raise PermissionError('Test mode cannot write checkpoints')
    if distributed.rank()==0:
        temporary = Path(str(filename) + '.tmp')
        torch.save(state, temporary)
        temporary.replace(filename)


def restore(filename, model, optimizer, scheduler, scaler, loader, cfg, validation_loader=None):
    from BASELINES.common import distributed
    state = filename if isinstance(filename,dict) else load_file(filename)
    required = {'model', 'config', 'optimizer', 'scheduler', 'scaler', 'rng', 'loader_rng',
                'history', 'epoch', 'step', 'best'}
    if not required.issubset(state):
        raise ValueError(f'Full resume requires an unfinished resumable checkpoint; missing state: {sorted(required-set(state))}')
    saved = state['config']
    # An epoch boundary is the resume boundary. Changing optimizer/data/model/budget
    # would no longer be a continuation of the same experiment.
    for key in ('model', 'data', 'optimization', 'seed', 'stage', 'route', 'precision',
                'max_batches', 'selection_metric'):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f'Resume configuration differs in {key}')
    if saved.get('data_audit', {}).get('manifests') != cfg.get('data_audit', {}).get('manifests'):
        raise ValueError('Resume manifests changed since the checkpoint was saved')
    if saved.get('data_audit', {}).get('split_policy') != cfg.get('data_audit', {}).get('split_policy'):
        raise ValueError('Resume CSV split policy differs from the checkpoint')
    roots = saved.get('data_audit', {}).get('media_roots')
    if roots is not None and roots != cfg.get('data_audit', {}).get('media_roots'):
        raise ValueError('Resume media roots changed since the checkpoint was saved')
    model.load_state_dict(state['model'], strict=True)
    if saved.get('gradient_check'):
        model._gradient_check=saved['gradient_check']
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    if state.get('rank_states'):
        if len(state['rank_states'])!=distributed.world_size():
            raise ValueError('Full resume requires the original distributed world size')
        state.update(state['rank_states'][distributed.rank()])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        if state['rng'].get('cuda_local_only'):torch.cuda.set_rng_state(state['rng']['cuda'][0])
        else:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    loader.generator.set_state(state['loader_rng'])
    if validation_loader is not None and 'validation_loader_rng' in state:
        validation_loader.generator.set_state(state['validation_loader_rng'])
    elif validation_loader is not None and int(cfg['data'].get('num_workers', 0)) and cfg['data'].get('sentence_eval_stride') == -1:
        raise ValueError('Legacy checkpoint lacks validation-worker RNG for random sentence evaluation')
    return state


def move(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move(v, device) for k, v in batch.items()}
    return batch


def check_evaluation_protocol(checkpoint, cfg):
    saved = checkpoint['config']
    keys = ('task', 'label_column', 'num_class', 'datasets', 'num_sentence', 'sentence_eval_stride',
            'cohort', 'video_column', 'eval_clips')
    for key in keys:
        if saved['data'].get(key) != cfg['data'].get(key):
            raise ValueError(f'Evaluation protocol differs from training: data.{key}')
    if saved.get('native_input') != cfg.get('native_input'):
        raise ValueError('Evaluation native model input differs from training')
    if saved['model'].get('instance_aggregator') != cfg['model'].get('instance_aggregator'):
        raise ValueError('Evaluation instance aggregation differs from training')
    for key in ('manifests', 'media_roots', 'split_policy'):
        if saved['data_audit'].get(key) is not None and saved['data_audit'][key] != cfg['data_audit'].get(key):
            raise ValueError(f'Evaluation {key} changed since training')


@contextmanager
def evaluation_rng(loader):
    """Repeat native masks/views and worker seeds without changing training RNGs."""
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    from BASELINES.common import distributed
    gpu_rng = ([torch.cuda.get_rng_state()] if distributed.active() else torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else []
    loader_rng = loader.generator.get_state()
    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    loader.generator.manual_seed(1701)
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if gpu_rng:
            if distributed.active():torch.cuda.set_rng_state(gpu_rng[0])
            else:torch.cuda.set_rng_state_all(gpu_rng)
        loader.generator.set_state(loader_rng)


def input_summary(raw, prepared, output):
    result = dict(raw_shapes={k:list(raw[k].shape) for k in ('audio', 'video') if k in raw},
                  model_input_shapes={k:list(v.shape) for k,v in prepared.items() if torch.is_tensor(v)},
                  output_shape=list(output.shape))
    if 'face_video_shape' in raw:result['raw_shapes']['face_before_mouth_crop']=raw['face_video_shape'].tolist()
    result['valid_lengths'] = {k:raw[k].tolist() for k in ('audio_lengths', 'video_lengths', 'video_valid_frames') if k in raw}
    result['timing'] = {k:float(raw[k]) for k in ('source_fps', 'start_time', 'end_time') if k in raw}
    for key in ('valid', 'attention_mask', 'padding_mask'):
        if key in prepared:
            value = prepared[key]
            value = value if key=='valid' else (value if key=='attention_mask' else ~value).sum(-1)
            result['valid_lengths'][key] = value.detach().cpu().tolist()
    return result


def update(loss, model, optimizer, scheduler, scaler, clip=1., accumulation=1, perform_step=True):
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite loss: {loss.item()}')
    scaler.scale(loss/accumulation).backward()
    if not perform_step:
        return None
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if norm == 0 and not hasattr(model, '_gradient_check'):
        raise RuntimeError('No nonzero gradient reached trainable parameters')
    probes = {}
    if not hasattr(model, '_gradient_check'):
        modules = {key: getattr(model, key) for key in ('encoder', 'head', 'projector') if hasattr(model, key)}
        if hasattr(model, 'gradient_components'):
            modules.update(model.gradient_components())
        if getattr(model, 'pretraining', False):
            for name in ('decoder_pred', 'decoder_pred_a', 'decoder', 'shared_decoder'):
                decoder = getattr(model.encoder, name, None)
                if isinstance(decoder, torch.nn.Module):
                    modules['pretraining_decoder'] = decoder
                    break
        for key, module in modules.items():
            candidates = [(n, p) for n, p in module.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
            if not candidates:
                raise RuntimeError(f'No nonzero gradient reached {key}')
            # Prefer an input projection to catch a detached encoder API.
            name, parameter = next(((n, p) for n, p in candidates if any(
                part in n for part in ('patch_embed', 'feature_extractor', 'extract_audio', 'local_encoder'))), candidates[0])
            index = int(parameter.grad.abs().reshape(-1).argmax())
            probes[key] = (name, parameter, index, parameter.detach().reshape(-1)[index].clone())
    scaler.step(optimizer)
    scaler.update()
    if probes:
        model._gradient_check = {}
        for key, (name, parameter, index, before) in probes.items():
            delta = float((parameter.detach().reshape(-1)[index]-before).abs())
            if delta == 0:
                raise RuntimeError(f'{key}.{name} received gradients but no parameter update')
            model._gradient_check[key] = dict(parameter=name, absolute_update=delta)
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def fit(model, loaders, cfg, output, logger, device, resume, epoch_fn):
    """Common epoch-boundary lifecycle; stages supply only their real objectives."""
    from BASELINES.common import distributed
    epochs = int(cfg['optimization'].get('epochs', 20))
    if not cfg.get('no_save'):
        import shutil
        # Include atomic replacement alongside the existing checkpoint. Native
        # SSL teachers/decoders can add storage beyond trainable parameters.
        tensors=sum(p.numel()*p.element_size() for p in model.parameters())
        needed=tensors*8
        free=shutil.disk_usage(output).free
        if free<needed:raise OSError(f'Insufficient checkpoint space: need about {needed/2**30:.1f} GiB free, have {free/2**30:.1f} GiB')
    batches = min(len(loaders['train']), int(cfg.get('max_batches', len(loaders['train']))))
    if epochs < 1 or batches < 1:
        raise ValueError('epochs and max_batches must be positive')
    accumulation = int(cfg['optimization'].get('gradient_accumulation', 1))
    optimizer, scheduler = optimizer_and_scheduler(model, cfg, epochs * math.ceil(batches/accumulation))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg.get('precision') == 'float16')
    start, step, best, history = 0, 0, None, []
    if resume:
        state = restore(resume, model, optimizer, scheduler, scaler, loaders['train'], cfg, loaders.get('validation'))
        start, step, best, history = state['epoch'], state['step'], state['best'], state['history']
        logger.info('restored epoch=%d step=%d, optimizer/scheduler/scaler/RNG/loader state', start, step)
    metric = cfg.get('selection_metric', 'loss' if cfg['stage'] == 'pretrain' else 'mae' if cfg['data']['task'] == 'regression' else 'f1_macro')
    maximize = metric not in ('loss', 'mae', 'mse', 'rmse')
    if 'validation' not in loaders:
        logger.info('CSV has no validation split: fixed epochs, final-epoch %s, no early stopping',
                    'latest.pt' if cfg['stage']=='pretrain' else 'best.pt')
    for epoch in range(start, epochs):
        if hasattr(loaders['train'].sampler, 'set_epoch'):
            loaders['train'].sampler.set_epoch(epoch)
        model.train()
        train = epoch_fn(loaders['train'], optimizer, scheduler, scaler, step)
        step += train.pop('updates')
        cfg['gradient_check'] = getattr(model, '_gradient_check', {})
        cfg['io_check'] = getattr(model, '_io_check', {})
        write_config(output, cfg)
        distributed.broadcast_batchnorm_buffers(model)
        model.eval()
        validation = None
        if 'validation' in loaders:
            with evaluation_rng(loaders['validation']), torch.no_grad():
                validation = epoch_fn(loaders['validation'], None, None, None, step)
            validation.pop('updates')
            score = validation[metric]
            if not math.isfinite(score):
                raise FloatingPointError(f'Nonfinite selection metric: {metric}={score}')
            improved = best is None or (score > best['score'] if maximize else score < best['score'])
            if improved:
                best = dict(epoch=epoch+1, score=score, metric=metric, validation=validation)
        else:
            improved = True
            best = dict(epoch=epoch+1, score=None, metric=None, validation=None, selection='final_epoch_no_csv_validation')
        history.append(dict(epoch=epoch+1, train=train, validation=validation))
        if not cfg.get('no_save') and (cfg['stage']=='pretrain' or improved):
            filename='latest.pt' if cfg['stage']=='pretrain' else 'best.pt'
            save_checkpoint(output / filename, model, optimizer, scheduler, scaler, loaders['train'], cfg, epoch+1, step, best, history,
                            validation_loader=loaders.get('validation'))
        report = dict(best=best, history=history,
                   scope='smoke_subset' if cfg['data'].get('limit') or cfg.get('max_batches') else 'full',
                   parameters=parameter_counts(model), optimizer_steps=step)
        if cfg.get('no_save'):
            cfg['test_result'] = report
        elif distributed.rank()==0:
            write_json(output / 'metrics.json', report)
        logger.info('epoch=%d/%d step=%d train_loss=%.6g selection=%s best_epoch=%d',
                    epoch+1, epochs, step, train['loss'], metric if validation else 'final_epoch', best['epoch'])
        if cfg.get('verify_resume') and epoch==0:
            memory=save_checkpoint(None,model,optimizer,scheduler,scaler,loaders['train'],cfg,epoch+1,step,best,history,
                                   validation_loader=loaders.get('validation'))
            import io
            with io.BytesIO() as stream:
                torch.save(memory,stream)
                del memory
                stream.seek(0)
                memory=torch.load(stream,map_location='cpu',weights_only=False)
            expected=(random.random(),float(np.random.rand()),torch.rand(3),torch.rand(3,generator=loaders['train'].generator))
            torch.manual_seed(9901)
            optimizer,scheduler=optimizer_and_scheduler(model,cfg,epochs*math.ceil(batches/accumulation))
            scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg.get('precision')=='float16')
            restore(memory,model,optimizer,scheduler,scaler,loaders['train'],cfg,loaders.get('validation'))
            assert random.random()==expected[0] and float(np.random.rand())==expected[1]
            assert torch.equal(torch.rand(3),expected[2]) and torch.equal(torch.rand(3,generator=loaders['train'].generator),expected[3])
            assert optimizer.state and scheduler.last_epoch==step
            cfg['resume_verified_in_memory']=True
            del memory
        if cfg.get('stop_after_epoch') and epoch+1 >= int(cfg['stop_after_epoch']):
            break
        patience = int(cfg['optimization'].get('early_stopping_patience', 10 if cfg['stage']=='finetune' else 0))
        if validation is not None and patience and epoch+1-best['epoch'] >= patience:
            logger.info('early stopping after %d validation events without improvement', patience)
            break
    return best
