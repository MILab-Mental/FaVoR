"""Audited initialization adapters and atomic VA checkpoint persistence."""
import logging
import os
from pathlib import Path
import torch
from models.video_lejepa import load_vjepa_encoder
from models.audio_lejepa import load_emotion2vec_encoder

LOGGER = logging.getLogger(__name__)
SCHEMA = 'favor.va_lejepa.v1'


def strip_ddp(state):
    return {key.removeprefix('module.'): value for key, value in state.items()}


def _load_lejepa(target, path, schema, prefix, weight_source):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('schema') != schema:
        raise ValueError(f'expected {schema}, got {checkpoint.get("schema")}')
    if weight_source not in {'encoder', 'ema'}:
        raise ValueError('weight_source must be encoder or ema')
    state = strip_ddp(checkpoint['encoder'])
    if weight_source == 'ema':
        shadow = checkpoint.get('state_dict_ema')
        if not isinstance(shadow, dict):
            raise ValueError('EMA requested but state_dict_ema is absent')
        shadow = strip_ddp(shadow)
        overrides = {key[len(prefix):]: value for key, value in shadow.items() if key.startswith(prefix)}
        expected = {key for key, value in state.items() if value.is_floating_point()}
        if set(overrides) != expected:
            raise ValueError(f'EMA encoder prefix {prefix!r} is incomplete or incompatible')
        state.update(overrides)
    expected = target.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [key for key in set(state) & set(expected) if state[key].shape != expected[key].shape]
    loaded = [key for key in expected if key in state and key not in mismatched]
    ratio = sum(expected[key].numel() for key in loaded) / max(sum(v.numel() for v in expected.values()), 1)
    report = dict(loaded=loaded, missing=missing, unexpected=unexpected, mismatched=mismatched,
                  loaded_ratio=ratio, weight_source=weight_source, source=str(path))
    LOGGER.info('VA encoder initialization: %s', report)
    if missing or unexpected or mismatched:
        raise ValueError(f'incompatible {schema} encoder: {report}')
    target.load_state_dict(state, strict=True)
    return report


def load_video_encoder_init(encoder, cfg):
    source = cfg.get('init_source', 'vjepa')
    if source == 'vjepa':
        if cfg.get('weight_source', 'encoder') != 'encoder':
            raise ValueError('V-JEPA init does not support LeJEPA EMA extraction')
        path = cfg['vjepa_checkpoint']
        if Path(path).exists():
            header = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if header.get('schema', '').startswith('favor.'):
                raise ValueError('V-JEPA initialization requires a V-JEPA checkpoint; choose init_source=video_lejepa for LeJEPA weights')
            del header
        report = load_vjepa_encoder(encoder, path, cfg.get('min_load_ratio', .90))
        return dict(report, init_source=source, weight_source='encoder', source=str(path))
    if source == 'video_lejepa':
        return _load_lejepa(encoder, cfg['video_lejepa_checkpoint'], 'favor.video_lejepa.v1',
                            'encoder.', cfg.get('weight_source', 'encoder'))
    raise ValueError(f'unknown video init_source: {source}')


def load_audio_encoder_init(encoder, cfg):
    source = cfg.get('init_source', 'audio_lejepa')
    if source == 'emotion2vec':
        if cfg.get('weight_source', 'encoder') != 'encoder':
            raise ValueError('emotion2vec init does not support LeJEPA EMA extraction')
        report = load_emotion2vec_encoder(encoder, cfg['emotion2vec_checkpoint'], cfg.get('min_load_ratio', .95))
        return dict(report, init_source=source, weight_source='encoder', source=str(cfg['emotion2vec_checkpoint']))
    if source == 'audio_lejepa':
        return _load_lejepa(encoder.backbone, cfg['audio_lejepa_checkpoint'], 'favor.audio_lejepa.v1',
                            'encoder.backbone.', cfg.get('weight_source', 'encoder'))
    raise ValueError(f'unknown audio init_source: {source}')


def save_checkpoint(path, model, optimizer=None, scaler=None, ema=None, step=0, epoch=0, config=None, **extra):
    raw = model.module if hasattr(model, 'module') else model
    va = raw.va if hasattr(raw, 'va') else raw
    state = dict(schema=SCHEMA, model=raw.state_dict(), video_encoder=va.video_encoder.state_dict(),
                 audio_encoder=va.audio_encoder.state_dict(), fusion_adapters=va.fusion_adapters.state_dict(),
                 fusion_input_adapters=va.input_adapters.state_dict(), fusion_loss_projectors=va.loss_projectors.state_dict(),
                 optional_unimodal_projectors=None, enabled_fusion_modes=list(va.enabled_modes),
                 alignment_config={'sample_rate': va.sample_rate, 'tubelet_size': 1},
                 initialization_report=va.initialization_report, config=config, step=int(step), epoch=int(epoch),
                 optimizer=optimizer.state_dict() if optimizer else None,
                 scaler=scaler.state_dict() if scaler else None, ema=ema.state_dict() if ema else None,
                 state_dict_ema=ema.state_dict()['shadow'] if ema else None, **extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, ema=None, *, allow_new_branches=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('not a VA-LeJEPA checkpoint')
    raw = model.module if hasattr(model, 'module') else model
    state = strip_ddp(checkpoint['model'])
    if allow_new_branches:
        target = raw.state_dict()
        # Selected existing branches must load completely; new ones alone may be absent.
        available = set(checkpoint['enabled_fusion_modes'])
        selected = set(raw.enabled_modes)
        state = {k: v for k, v in state.items() if not any(k.startswith(f'{family}.{mode}.')
                 for family in ('input_adapters', 'fusion_adapters', 'loss_projectors') for mode in available - selected)}
        missing = set(target) - set(state)
        allowed = {k for k in target if any(k.startswith(f'{family}.{mode}.')
                   for family in ('input_adapters', 'fusion_adapters', 'loss_projectors') for mode in selected - available)}
        if missing != allowed or set(state) - set(target):
            raise ValueError('checkpoint is missing existing branch/backbone weights')
        state = {**{k: target[k] for k in allowed}, **state}
        LOGGER.info('new randomly initialized VA branches: %s', sorted(selected - available))
    raw.load_state_dict(state, strict=True)
    va = raw.va if hasattr(raw, 'va') else raw
    va.initialization_report = checkpoint.get('initialization_report', {})
    for instance, key in ((optimizer, 'optimizer'), (scaler, 'scaler'), (ema, 'ema')):
        if instance is not None:
            if checkpoint.get(key) is None:
                if key == 'scaler' and not instance.is_enabled():
                    continue  # bf16/float32 have no dynamic loss-scaling state
                raise ValueError(f'full resume requires {key}')
            instance.load_state_dict(checkpoint[key])
    return checkpoint
