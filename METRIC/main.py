#!/usr/bin/env python3
"""FaVoR frozen-encoder probe and checkpoint selection experiment."""
from __future__ import annotations

# Keep every cache, temporary file and generated artifact in this directory.
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in ('results', 'logs', 'cache', 'cache/tmp'):
    (HERE / directory).mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TMPDIR'] = str(HERE / 'cache/tmp')
os.environ['XDG_CACHE_HOME'] = str(HERE / 'cache')
os.environ['MPLCONFIGDIR'] = str(HERE / 'cache/matplotlib')
os.environ['TORCH_HOME'] = str(HERE / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(HERE / 'cache/cuda')
os.environ['TRITON_CACHE_DIR'] = str(HERE / 'cache/triton')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(HERE / 'cache/torchinductor')
os.environ['NUMBA_CACHE_DIR'] = str(HERE / 'cache/numba')
os.environ['HF_HOME'] = str(HERE / 'cache/huggingface')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import yaml

from metrics import correlations, evaluate, leave_one_task_out, sample_split, transfer_scores
from datasets.common.instances import sentence_identity
from datasets.common.paths import dataset_roots

LOG = logging.getLogger('metric')
CATALOG = ROOT / 'CONFIGS/tasks/finetune/tasks.csv'
REGISTRY = ROOT / 'CONFIGS/datasets.csv'
FEATURE_VERSION = 3


def absolute(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def digest(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def file_signature(value):
    path = absolute(value).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def read_csv(path):
    with absolute(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def emotion_task_ids():
    # The catalog's nonmental group also includes mental-health datasets;
    # only datasets with the existing 2_ prefix are emotion datasets.
    return {row['task_id'] for row in read_csv(CATALOG) if row.get('dataset', '').startswith('2_')}


def tasks_in_scope(group, scope, emotion_tasks):
    if scope == 'all':
        return group
    if scope == 'emotion':
        return group[group['task_id'].isin(emotion_tasks)]
    return group[group['scope'] == scope]


def source_index(task, rule, registry, cache):
    """Return exact path->IDs and official-test IDs; never persist identifiers."""
    if 'pattern' in rule:
        pattern = re.compile(rule['pattern'])
        def lookup(row):
            match = pattern.search(Path(row.get('audio_path', '')).name)
            return match.group(1) if match else None
        return lookup, set()
    if 'source' not in rule or not rule.get('columns'):
        raise ValueError('subject rule needs source and columns, or a pattern')
    source = registry[task['csv_path']]['original_csv_path'] if rule['source'] == 'registry' else rule['source']
    key = (str(absolute(source)), tuple(rule['columns']))
    if key not in cache:
        mapping, official_test = defaultdict(set), set()
        with absolute(source).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'audio_path', *rule['columns']}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f'subject source lacks required columns: {sorted(required)}')
            for row in reader:
                path = (row.get('audio_path') or '').strip()
                fields = [(row.get(col) or '').strip() for col in rule['columns']]
                if not path or not all(fields):
                    continue
                subject = '\0'.join(fields)
                mapping[path].add(subject)
                original_split = str(row.get('split', '')).strip().lower()
                if re.fullmatch(r'test(?:\d+)?', original_split):
                    official_test.add(subject)
        cache[key] = mapping, official_test
    mapping, official_test = cache[key]
    def lookup(row):
        identifiers = mapping.get((row.get('audio_path') or '').strip(), set())
        return next(iter(identifiers)) if len(identifiers) == 1 else None
    return lookup, official_test


def parse_label(raw, task, count):
    if task == 'classification':
        label = int(raw) - 1
        if not 0 <= label < count:
            raise ValueError('classification label is outside the declared one-based range')
        return label
    if task == 'regression':
        label = float(raw)
        if not np.isfinite(label):
            raise ValueError('regression label is not finite')
        return label
    value = np.zeros(count, dtype=np.float32)
    for token in raw.split('|'):
        position = int(token.strip()) - 1
        if not 0 <= position < count:
            raise ValueError('multilabel index outside declared class range')
        value[position] = 1
    if not value.any():
        raise ValueError('empty multilabel target')
    return value


def protected_subjects(tasks, cfg, registry, source_cache):
    """Exclude specified downstream held-out people from every related task."""
    protected = defaultdict(set)
    by_name = {task['task_id']: task for task in tasks}
    for name in cfg.get('protected_tasks', []):
        if name not in by_name:
            raise ValueError(f'unknown protected task: {name}')
        task = by_name[name]
        rule = cfg.get('subject_rules', {}).get(task['dataset'])
        if not rule:
            raise ValueError(f'protected task lacks a subject identity rule: {name}')
        lookup, official_test = source_index(task, rule, registry, source_cache)
        heldout = protected[task['dataset']]
        heldout.update(official_test)
        split_column = task['label_column'] + '_split'
        rows = read_csv(task['csv_path'])
        if not rows or split_column not in rows[0]:
            raise ValueError(f'protected task split is missing: {name}')
        for row in rows:
            if str(row.get(split_column, '')).strip() in {'1', '2'}:
                subject = lookup(row)
                if subject is None:
                    raise ValueError(f'protected task has unresolved held-out subjects: {name}')
                heldout.add(subject)
        LOG.info('Protected %d held-out subjects for %s', len(heldout), name)
    return protected


def prepare_task(task, cfg, registry, source_cache, protected=None):
    dataset = task['dataset']
    rule = cfg.get('subject_rules', {}).get(dataset)
    if not rule:
        raise ValueError(f'no trusted subject identity rule for {dataset}')
    lookup, official_test = source_index(task, rule, registry, source_cache)
    rows = read_csv(task['csv_path'])
    split_column, label_column = task['label_column'] + '_split', task['label_column']
    if not rows or split_column not in rows[0] or label_column not in rows[0]:
        raise ValueError('task label or split column missing')
    heldout = set(official_test)
    heldout.update((protected or {}).get(dataset, set()))
    heldout_media = {'audio': set(), 'video': set()}
    root_value = registry.get(task['csv_path'], {}).get('root_path')
    root = Path(root_value) if root_value else None
    unresolved_heldout = 0
    for row in rows:
        if str(row.get(split_column, '')).strip() in {'1', '2'}:
            subject = lookup(row)
            if subject is None:
                unresolved_heldout += 1
            else:
                heldout.add(subject)
            if root:
                for modality in ('audio', 'video'):
                    value = (row.get(modality + '_path') or '').strip()
                    if value:
                        heldout_media[modality].add(str(root / value))
    if unresolved_heldout:
        raise ValueError(f'{unresolved_heldout} held-out rows lack unambiguous subject IDs')
    grouped, missing_id, excluded = {}, 0, 0
    for row in rows:
        if str(row.get(split_column, '')).strip() != '0':
            continue
        subject = lookup(row)
        if subject is None:
            missing_id += 1
            continue
        if subject in heldout:
            excluded += 1
            continue
        path = (row.get('audio_path') or '').strip()
        video = (row.get('video_path') or '').strip()
        raw_label = (row.get(label_column) or '').strip()
        if not path or not raw_label or (cfg['modality'] != 'audio' and not video):
            continue
        label = parse_label(raw_label, task['task_type'], int(task['num_class']))
        # The project's recording identity groups -sNNNN sentence files.
        record, position = sentence_identity(path) if task['profile'] == 'instance' else (path, 0)
        if record not in grouped:
            grouped[record] = {'record': record, 'subject': subject, 'label': label, 'items': []}
        example = grouped[record]
        if example['subject'] != subject or not np.array_equal(example['label'], label):
            raise ValueError('a recording has inconsistent labels or subject IDs')
        example['items'].append((position, path, video, raw_label))
    examples = list(grouped.values())
    if not examples:
        raise ValueError(f'no train examples remain after subject/test exclusion (excluded rows={excluded})')
    for example in examples:
        example['items'].sort(key=lambda item: (item[0], item[1]))
        if task['profile'] == 'instance':
            ids = np.rint(np.linspace(0, len(example['items']) - 1, cfg['num_sentences'])).astype(int)
            example['items'] = [example['items'][i] for i in ids]
        else:
            example['items'] = example['items'][:1]
    y = np.asarray([e['label'] for e in examples])
    subjects = np.asarray([e['subject'] for e in examples])
    seed = (int(cfg['seed']) + int(digest(task['task_id'])[:8], 16)) % (2**32 - 1)
    max_samples = int(cfg.get('samples_per_task', 0) or 0)
    if max_samples < 0:
        raise ValueError('samples_per_task must be nonnegative')
    train, val = sample_split(y, subjects, task['task_type'], int(cfg['samples_per_class']),
                              float(cfg['val_fraction']), int(cfg['regression_bins']), seed,
                              max_samples=max_samples or None)
    assert not (set(subjects[train]) & set(subjects[val]))
    assert not ((set(subjects[train]) | set(subjects[val])) & heldout)
    selected = [examples[i] for i in np.concatenate([train, val])]
    membership = ['train'] * len(train) + ['val'] * len(val)
    signature = digest(json.dumps([(e['record'], e['subject'], e['items'], part)
                                   for e, part in zip(selected, membership)], ensure_ascii=False))
    audit = [{'task_id': task['task_id'], 'split': part,
              'subject_hash': digest(dataset + '\0' + e['subject']),
              'record_hash': digest(dataset + '\0' + e['record'])}
             for e, part in zip(selected, membership)]
    return {'task': task, 'examples': selected, 'partition': membership, 'n_train': len(train),
            'audit': audit, 'signature': signature, 'missing_id': missing_id,
            'excluded_heldout_rows': excluded, 'heldout_media': heldout_media, 'seed': seed}


def audited_load(model, state, source):
    """Do not silently score a partially/randomly initialized encoder."""
    expected = model.state_dict()
    cleaned = {}
    for key, value in state.items():
        while key.startswith(('module.', 'model.', 'backbone.')):
            key = key.split('.', 1)[1]
        cleaned[key] = value
    compatible = {k: v for k, v in cleaned.items() if k in expected and v.shape == expected[k].shape}
    missing_parameters = [k for k, _ in model.named_parameters() if k not in compatible]
    if missing_parameters:
        raise ValueError(f'{source}: incomplete encoder, missing {len(missing_parameters)} parameters: {missing_parameters[:5]}')
    model.load_state_dict(compatible, strict=False)
    LOG.info('Encoder checkpoint audited: %d/%d tensors from %s', len(compatible), len(expected), source)


def model_config(checkpoint):
    return checkpoint.get('config') or checkpoint.get('args') or {}


def checkpoint_metadata(checkpoint):
    saved = model_config(checkpoint)
    data = saved.get('data', {})
    manifests = data.get('manifests', data.get('datasets', []))
    if isinstance(manifests, str):
        manifests = [manifests]
    return {'checkpoint_schema': checkpoint.get('schema') or 'legacy',
            'checkpoint_epoch': checkpoint.get('epoch'),
            'checkpoint_step': checkpoint.get('step', checkpoint.get('global_step')),
            'pretrain_manifests': list(manifests or [])}


def metadata_cache_path(ckpt):
    key = digest(json.dumps(file_signature(ckpt['path']), sort_keys=True))
    return HERE / 'cache' / ('checkpoint-' + key[:24] + '.json')


def saved_checkpoint_metadata(ckpt, value=None):
    path = metadata_cache_path(ckpt)
    if value is not None:
        path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding='utf-8')
        return value
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    checkpoint = torch.load(absolute(ckpt['path']), map_location='cpu', weights_only=False, mmap=True)
    value = checkpoint_metadata(checkpoint)
    del checkpoint
    return saved_checkpoint_metadata(ckpt, value)


def build_model(checkpoint_path, modality, cfg, device):
    from app.main import load_config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    schema = checkpoint.get('schema', '')
    saved = model_config(checkpoint)
    if modality == 'audio':
        from models.audio_jepa.backbone import AudioBackbone
        from utils.audio_checkpoint import audio_backbone_state, load_official_emotion2vec_backbone
        if schema == 'favor.va_lejepa.v1':
            model_cfg = copy.deepcopy(saved['model']['audio'])
            state = {k.removeprefix('backbone.'): v for k, v in checkpoint['audio_encoder'].items()}
        else:
            model_cfg = copy.deepcopy(saved.get('model') or load_config(ROOT / 'CONFIGS/tasks/finetune/defaults/a/clip/default.yaml')['model'])
            state = audio_backbone_state(checkpoint)
        model_cfg['audio'] = {'sample_rate': 16000, 'process_seconds': 4., 'max_process_seconds': 4.}
        model = AudioBackbone(model_cfg)
        raw = checkpoint.get('model', {})
        if isinstance(raw, dict) and any(k.startswith('d2v_model.blocks.') for k in raw):
            report = load_official_emotion2vec_backbone(model, raw, min_parameter_ratio=1.0)
            if report.get('missing_targets'):
                raise ValueError('official audio checkpoint leaves encoder parameters uninitialized')
        else:
            audited_load(model, state, str(checkpoint_path))
        family = 'audio'
    elif modality == 'video':
        from models.video_jepa import vision_transformer as vit
        model_cfg = saved.get('model', {})
        data = saved.get('data', {})
        if schema == 'favor.video_lejepa.v1':
            from models.video_lejepa.encoder import build_encoder
            model = build_encoder(model_name=model_cfg.get('name', 'vit_large'),
                img_size=int(cfg['video_size']), num_frames=int(cfg['video_frames']),
                patch_size=int(model_cfg.get('patch_size', 16)), tubelet_size=int(model_cfg.get('tubelet_size', 1)),
                use_rope=bool(model_cfg.get('use_rope', True)), use_sdpa=bool(model_cfg.get('use_sdpa', True)),
                token_drop_rate=0, attn_mode=model_cfg.get('attn_mode', 'block_causal'),
                use_activation_checkpointing=False)
            state = checkpoint['encoder']
            family = 'video_lejepa'
        else:
            default = load_config(ROOT / 'CONFIGS/tasks/finetune/defaults/v/clip/default.yaml')
            spec, defaults = default['model'], default['data']
            model = vit.__dict__[spec['model_name']](img_size=int(cfg['video_size']),
                patch_size=int(defaults.get('patch_size', 16)), num_frames=int(cfg['video_frames']),
                tubelet_size=int(defaults.get('tubelet_size', 2)), out_layers=spec.get('out_layers', [23]),
                uniform_power=spec.get('uniform_power', True), use_rope=spec.get('use_rope', True),
                use_sdpa=spec.get('use_sdpa', True), use_activation_checkpointing=False)
            state = checkpoint.get('encoder', checkpoint.get('target_encoder', checkpoint))
            family = 'video_jepa'
        audited_load(model, state, str(checkpoint_path))
    elif modality == 'audio-video':
        from models.va_lejepa.model import build_va_lejepa
        from models.va_lejepa.checkpoint import strip_ddp
        if schema != 'favor.va_lejepa.v1':
            raise ValueError('audio-video evaluation requires a VA-LeJEPA checkpoint')
        data = copy.deepcopy(saved['data'])
        model_cfg = copy.deepcopy(saved['model'])
        data.update(video_global_size=int(cfg['video_size']), video_frames_global=int(cfg['video_frames']),
                    video_fps=int(cfg['video_fps']), global_seconds=int(cfg['video_frames']) / int(cfg['video_fps']))
        model_cfg['audio_gradient_checkpointing'] = False
        model_cfg['video']['use_activation_checkpointing'] = False
        model = build_va_lejepa(model_cfg, data, initialize=False)
        model.load_state_dict(strip_ddp(checkpoint['model']), strict=True)
        if cfg['va_branch'] not in model.enabled_modes:
            raise ValueError('requested VA branch was not trained in this checkpoint')
        family = 'audio-video'
    else:
        raise ValueError('modality must be audio, video, or audio-video')
    metadata = checkpoint_metadata(checkpoint)
    del checkpoint
    gc.collect()
    model.requires_grad_(False).eval().to(device)
    return model, family, metadata


def audit_pretraining(prepared, checkpoint, metadata, modality, cache):
    """Compare held-out paths to the *current* source; historical input may differ."""
    from datasets.audio_jepa.pretrain_dataset import _manifest_path
    manifests = metadata.get('pretrain_manifests', [])
    rows = []
    kinds = ('audio', 'video') if modality == 'audio-video' else (modality,)
    if not manifests:
        return [{'task_id': p['task']['task_id'], 'checkpoint': checkpoint['id'], 'manifest': '',
                 'modality': modality, 'status': 'manifest_not_recorded', 'heldout_paths': np.nan,
                 'exact_path_matches': np.nan, 'manifest_newer_than_checkpoint': np.nan} for p in prepared]
    for manifest in manifests:
        path = absolute(manifest)
        if not path.is_file():
            for p in prepared:
                for kind in kinds:
                    rows.append(dict(task_id=p['task']['task_id'], checkpoint=checkpoint['id'],
                        manifest=str(path), modality=kind, status='manifest_unavailable', heldout_paths=len(p['heldout_media'][kind]),
                        exact_path_matches=np.nan, manifest_newer_than_checkpoint=np.nan))
            continue
        signature = json.dumps(file_signature(path), sort_keys=True) + '\0' + modality
        if signature not in cache:
            # One streaming pass regardless of how many tasks/checkpoints share this manifest.
            candidates = {kind: defaultdict(set) for kind in kinds}
            for p in prepared:
                for kind in kinds:
                    for value in p['heldout_media'][kind]:
                        candidates[kind][value].add(p['task']['task_id'])
            hits = defaultdict(set)
            with path.open(encoding='utf-8-sig', newline='') as handle:
                if modality == 'audio-video':
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or not {'audio_path', 'video_path'}.issubset(reader.fieldnames):
                        raise ValueError('paired pretraining manifest lacks audio/video columns')
                    stream = ((kind, row.get(kind + '_path', '')) for row in reader for kind in kinds)
                else:
                    stream = ((modality, _manifest_path(line)) for line in handle)
                for kind, value in stream:
                    if value in candidates[kind]:
                        for task_id in candidates[kind][value]:
                            hits[(task_id, kind)].add(value)
            cache[signature] = {key: len(values) for key, values in hits.items()}
        newer = path.stat().st_mtime_ns > absolute(checkpoint['path']).stat().st_mtime_ns
        for p in prepared:
            for kind in kinds:
                rows.append(dict(task_id=p['task']['task_id'], checkpoint=checkpoint['id'],
                    manifest=str(path), modality=kind, status='current_manifest_only',
                    heldout_paths=len(p['heldout_media'][kind]),
                    exact_path_matches=cache[signature].get((p['task']['task_id'], kind), 0),
                    manifest_newer_than_checkpoint=newer))
    return rows


def make_dataset(prepared, cfg):
    task = prepared['task']
    root = dataset_roots()[str(absolute(task['csv_path']).resolve())]
    common = dict(label_column=task['label_column'], task=task['task_type'], num_class=int(task['num_class']))
    if cfg['modality'] == 'audio':
        from datasets.audio_jepa.finetune_dataset import AudioCSVDataset
        ds = AudioCSVDataset([str(absolute(task['csv_path']))], 0, root_paths=[root],
            sample_rate=16000, process_seconds=4., num_clips=int(cfg['num_audio_clips']), training=False, **common)
        index = {str(Path(path).resolve()): i for i, path in enumerate(ds.samples)}
    elif cfg['modality'] == 'video':
        from datasets.video_jepa.finetune_dataset import VideoCSVDataset
        from datasets.video_jepa.transforms import make_finetune_transforms
        ds = VideoCSVDataset([str(absolute(task['csv_path']))], 0, root_paths=[root],
            frames_per_clip=int(cfg['video_frames']), fps=int(cfg['video_fps']), frame_step=None,
            num_clips=1, random_clip_sampling=False,
            transform=make_finetune_transforms(False, int(cfg['video_size'])), **common)
        index = {str(Path(path).resolve()): i for i, path in enumerate(ds.samples)}
    else:
        from datasets.va_lejepa.finetune_dataset import AVCSVDataset
        # The existing VA loader places its index next to its CSV, so its input
        # is a small selected-only manifest under METRIC/cache, never DATASET/.
        manifest = HERE / 'cache' / ('va-' + prepared['signature'][:20] + '.csv')
        entries = {}
        for example in prepared['examples']:
            for _, audio, video, raw in example['items']:
                entries[audio] = {'audio_path': audio, 'video_path': video,
                                  task['label_column']: raw, task['label_column'] + '_split': '0'}
        try:
            existing = read_csv(manifest) if manifest.exists() else []
        except (OSError, csv.Error):
            existing = []
        if existing != list(entries.values()):
            temporary = manifest.with_suffix('.tmp')
            with temporary.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['audio_path', 'video_path', task['label_column'], task['label_column'] + '_split'])
                writer.writeheader()
                writer.writerows(entries.values())
            os.replace(temporary, manifest)
        seconds = int(cfg['video_frames']) / int(cfg['video_fps'])
        data = dict(datasets=[str(manifest)], root=root, sample_rate=16000, task=task['task_type'],
            label_column=task['label_column'], num_class=int(task['num_class']), local_views=0,
            global_seconds=seconds, local_seconds=min(2., seconds), video_fps=int(cfg['video_fps']),
            video_frames_global=int(cfg['video_frames']), video_frames_local=int(cfg['video_fps'] * min(2., seconds)),
            video_global_size=int(cfg['video_size']), short_policy='zero_pad', minimum_seconds=.1,
            normalize_audio=True, on_decode_error='error', sync=dict(strict=True, require_same_source_id=True,
                require_same_start=True, require_same_duration=True, duration_tolerance_seconds=.5,
                duration_tolerance_frames=2, offset_tolerance_seconds=.1, on_mismatch='error'))
        ds = AVCSVDataset(data, 0, training=False)
        index = {str(Path(row['audio_path']).resolve()): i for i, row in enumerate(ds.rows)}
    return ds, index, Path(root)


def extract(prepared, model, family, cfg, device):
    from datasets.va_lejepa.collate import collate_views, to_device
    if '_dataset' not in prepared:
        prepared['_dataset'] = make_dataset(prepared, cfg)
    ds, index, root = prepared['_dataset']
    examples = prepared['examples']
    output = []
    batch_size = int(cfg['batch_size'])
    def decode(spec):
        example, (_, path, video, _) = spec
        relative = video if cfg['modality'] == 'video' else path
        key = str((root / relative).resolve())
        if key not in index:
            raise ValueError('sampled media path not found in filtered dataset')
        try:
            item = ds.get_sentence(index[key])
        except Exception as error:
            raise ValueError(f'decode error for record {digest(example["record"])[:16]} '
                             f'({type(error).__name__})') from error
        if family == 'audio':
            return item[0]
        if family.startswith('video'):
            return item[0][0][0]
        if 'decode_error' in item:
            raise ValueError('selected paired media failed decoding')
        return item['global']

    workers = int(cfg.get('decode_workers', 1)) if family.startswith('video') else 1
    with ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext() as executor:
        for begin in range(0, len(examples), batch_size):
            batch = examples[begin:begin + batch_size]
            counts = [len(example['items']) for example in batch]
            specs = [(example, item) for example in batch for item in example['items']]
            inputs = list(executor.map(decode, specs)) if executor else [decode(spec) for spec in specs]
            mixed = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with torch.inference_mode(), mixed:
                if family == 'audio':
                    tensor = torch.stack(inputs).to(device)
                    segments, clips, length = tensor.shape
                    feature = model.extract_features(tensor.reshape(segments * clips, length), pool=True)
                    feature = feature.reshape(segments, clips, -1).mean(1)
                elif family.startswith('video'):
                    tensor = torch.stack(inputs).to(device)
                    feature = model(tensor, return_tokens=False) if family == 'video_lejepa' else model(tensor)
                    if isinstance(feature, (tuple, list)):
                        # Match ClipEncoder's equal-token pooling over every
                        # configured output layer in the full fine-tuning run.
                        feature = torch.cat(feature, dim=1)
                    if feature.ndim == 3:
                        feature = feature.mean(1)
                else:
                    view = to_device(collate_views(inputs), device)
                    feature = model.encode_view(view)[cfg['va_branch']]
            features = feature.float().cpu().numpy()
            offset = 0
            for count in counts:
                output.append(features[offset:offset + count].mean(0))
                offset += count
    features = np.stack(output)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite extracted features')
    y = np.asarray([e['label'] for e in examples])
    return features, y


def feature_cache_path(prepared, ckpt, cfg, device):
    settings = {k: cfg[k] for k in ('modality', 'num_sentences', 'num_audio_clips', 'video_frames', 'video_fps', 'video_size', 'va_branch')}
    settings.update(batch_size=int(cfg['batch_size']), device_type=device.type, torch_version=str(torch.__version__))
    if cfg['modality'] == 'video':
        settings['video_readout'] = 'mean-all-output-layers-v1'
    root = Path(dataset_roots()[str(absolute(prepared['task']['csv_path']).resolve())])
    media = set()
    for example in prepared['examples']:
        for _, audio, video, _ in example['items']:
            if cfg['modality'] in ('audio', 'audio-video'):
                media.add(str(root / audio))
            if cfg['modality'] in ('video', 'audio-video'):
                media.add(str(root / video))
    signature = digest(json.dumps({'v': FEATURE_VERSION, 'samples': prepared['signature'],
        'checkpoint': file_signature(ckpt['path']), 'settings': settings,
        'media': [file_signature(path) for path in sorted(media)]}, sort_keys=True))
    return HERE / 'cache' / ('features-' + signature[:24] + '.npz')


def cached_features(prepared, ckpt, model, family, cfg, device):
    filename = feature_cache_path(prepared, ckpt, cfg, device)
    if filename.exists():
        with np.load(filename, allow_pickle=False) as data:
            x, y = data['features'], data['targets']
        expected = np.asarray([e['label'] for e in prepared['examples']])
        if (len(x) == len(prepared['examples']) and len(y) == len(x)
                and np.isfinite(x).all() and np.array_equal(y, expected)):
            return x, y
        raise ValueError('feature cache failed shape/finite validation')
    if model is None:
        raise ValueError('feature cache missing; run once without --metrics-only')
    x, y = extract(prepared, model, family, cfg, device)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=filename.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, features=x, targets=y)
        os.replace(temporary, filename)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return x, y


def summarize(raw):
    rows = []
    emotion_tasks = emotion_task_ids()
    for (trajectory, modality), experiment in raw.groupby(['trajectory', 'modality']):
        groups = list(experiment.groupby('checkpoint'))
        common = set.intersection(*(set(group['task_id']) for _, group in groups))
        for checkpoint, group in groups:
            group = group[group['task_id'].isin(common)]
            for scope in ('mental', 'emotion', 'nonmental', 'all'):
                part = tasks_in_scope(group, scope, emotion_tasks)
                scores = part['transfer_score'].dropna()
                if len(scores):
                    rows.append(dict(trajectory=trajectory, modality=modality, checkpoint=checkpoint,
                                     scope=scope, tasks=len(scores), transfer_score=float(scores.mean())))
    return pd.DataFrame(rows, columns=['trajectory', 'modality', 'checkpoint', 'scope', 'tasks', 'transfer_score'])


def load_finetune(cfg):
    required = {'trajectory', 'modality', 'task_id', 'checkpoint', 'finetune_metric',
                'finetune_mode', 'finetune_split', 'finetune_value'}
    if cfg.get('finetune_csv'):
        ft = pd.read_csv(absolute(cfg['finetune_csv']))
        if not required.issubset(ft.columns):
            raise ValueError(f'fine-tuning CSV is missing columns: {sorted(required - set(ft.columns))}')
    else:
        frames = []
        for source in cfg.get('finetune_sources', []):
            path = absolute(source['path'])
            if not path.is_file():
                LOG.warning('Optional fine-tuning source unavailable: %s', path)
                continue
            table = pd.read_csv(path)
            checkpoint_col, value_col = source['checkpoint_column'], source['value_column']
            if checkpoint_col not in table or value_col not in table:
                raise ValueError(f'fine-tuning source {path} lacks checkpoint or value column')
            converted = pd.DataFrame({
                'trajectory': source['trajectory'], 'modality': source['modality'],
                'task_id': source['task_id'],
                'checkpoint': source.get('checkpoint_prefix', '') + table[checkpoint_col].astype(str),
                'finetune_metric': source['finetune_metric'], 'finetune_mode': source['finetune_mode'],
                'finetune_split': source['finetune_split'], 'finetune_value': table[value_col],
            })
            protocol_columns = source.get('protocol_columns', [])
            if any(column not in table for column in protocol_columns):
                raise ValueError(f'fine-tuning source {path} lacks protocol columns')
            converted['finetune_protocol'] = table[protocol_columns].astype(str).agg('/'.join, axis=1) if protocol_columns else 'unverified'
            if source.get('checkpoint_directory'):
                directory = absolute(source['checkpoint_directory']).resolve()
                eligible = {item['id'] for item in cfg['checkpoints']
                            if absolute(item['path']).resolve() == directory / (item['id'] + '.pt')}
                converted = converted[converted['checkpoint'].isin(eligible)]
            frames.append(converted)
        ft = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=sorted(required))
    if not ft['finetune_mode'].isin(['min', 'max']).all():
        raise ValueError('finetune_mode must be min or max')
    if 'finetune_protocol' not in ft:
        ft['finetune_protocol'] = 'unverified'
    ft['checkpoint'] = ft['checkpoint'].astype(str)
    ft['finetune_value'] = pd.to_numeric(ft['finetune_value'], errors='raise')
    return ft


def summarize_correlations(corr):
    columns = ['trajectory', 'modality', 'scope', 'metric', 'finetune_metric', 'finetune_split', 'protocol_consistency',
               'tasks_with_correlation', 'mean_spearman', 'mean_kendall',
               'tasks_with_regret', 'mean_normalized_regret', 'pairwise_tasks',
               'pairwise_correct_fraction', 'pairwise_mean_regret', 'pairwise_metric_ties']
    rows = []
    emotion_tasks = emotion_task_ids()
    if not corr.empty:
        keys = ['trajectory', 'modality', 'metric', 'finetune_metric', 'finetune_split', 'protocol_consistency']
        for key, group in corr.groupby(keys, dropna=False):
            for scope in ('mental', 'emotion', 'nonmental', 'all'):
                part = tasks_in_scope(group, scope, emotion_tasks)
                if part.empty:
                    continue
                pair = part[(part['pairs'] == 2) & (~part['metric_tie']) & (~part['best_tie'])]
                rows.append(dict(zip(keys, key), scope=scope,
                    tasks_with_correlation=int(part['spearman'].notna().sum()),
                    mean_spearman=part['spearman'].mean(), mean_kendall=part['kendall'].mean(),
                    tasks_with_regret=int(part['normalized_regret'].notna().sum()),
                    mean_normalized_regret=part['normalized_regret'].mean(),
                    pairwise_tasks=len(pair), pairwise_correct_fraction=(pair['selection_regret'] <= 1e-12).mean() if len(pair) else np.nan,
                    pairwise_mean_regret=pair['selection_regret'].mean() if len(pair) else np.nan,
                    pairwise_metric_ties=int(((part['pairs'] == 2) & part['metric_tie']).sum())))
    return pd.DataFrame(rows, columns=columns)


def fixed_checkpoint_baselines(raw, ft):
    """Score the same checkpoint on every eligible task as a simple baseline."""
    columns = ['trajectory', 'modality', 'scope', 'finetune_metric', 'finetune_split',
               'protocol_consistency', 'checkpoint', 'tasks', 'correct_fraction', 'mean_regret']
    if ft.empty:
        return pd.DataFrame(columns=columns)
    keys = ['trajectory', 'modality', 'task_id', 'checkpoint']
    merged = raw[keys + ['scope']].merge(ft, on=keys, how='inner', validate='one_to_one')
    rows = []
    emotion_tasks = emotion_task_ids()
    for _, part in merged.groupby(['trajectory', 'modality', 'task_id'], dropna=False):
        if (len(part) < 2 or part['finetune_mode'].nunique() != 1 or
                part['finetune_metric'].nunique() != 1 or part['finetune_split'].nunique() != 1):
            continue
        scores = part['finetune_value'].to_numpy(float) * (1 if part['finetune_mode'].iloc[0] == 'max' else -1)
        if not np.isfinite(scores).all() or np.isclose(scores.max(), scores.min(), rtol=1e-12, atol=1e-12):
            continue
        protocols = part['finetune_protocol']
        consistency = ('unverified' if protocols.isna().any() or (protocols == 'unverified').any() else
                       ('consistent' if protocols.nunique() == 1 else 'mixed'))
        for (_, row), score in zip(part.iterrows(), scores):
            scopes = [row['scope'], 'all'] + (['emotion'] if row['task_id'] in emotion_tasks else [])
            for scope in scopes:
                rows.append(dict(trajectory=row['trajectory'], modality=row['modality'], scope=scope,
                    finetune_metric=row['finetune_metric'], finetune_split=row['finetune_split'],
                    protocol_consistency=consistency, checkpoint=row['checkpoint'], regret=float(scores.max() - score)))
    if not rows:
        return pd.DataFrame(columns=columns)
    keys = columns[:-3]
    data = pd.DataFrame(rows)
    return data.groupby(keys, dropna=False)['regret'].agg(
        tasks='size', correct_fraction=lambda x: (x <= 1e-12).mean(), mean_regret='mean').reset_index()[columns]


def plot_scores(summary, result_dir):
    if summary.empty:
        (result_dir / 'transfer_scores.png').unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scope, part in summary.groupby('scope', sort=False):
        ax.plot(part['checkpoint'], part['transfer_score'], marker='o', label=scope)
    ax.set(xlabel='Checkpoint', ylabel='Mean within-task transfer score', ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(result_dir / 'transfer_scores.png', dpi=140)
    plt.close(fig)


def plot_correlations(summary, result_dir):
    available = summary[(summary['scope'] == 'all') & summary['mean_spearman'].notna()]
    if available.empty:
        (result_dir / 'correlations.png').unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f'{r.trajectory} / {r.metric} (n={r.tasks_with_correlation})' for r in available.itertuples()]
    fig, ax = plt.subplots(figsize=(8, max(3, .35 * len(labels))))
    ax.barh(labels, available['mean_spearman'])
    ax.set(xlabel='Mean within-task Spearman correlation', xlim=(-1, 1))
    ax.axvline(0, color='black', linewidth=.7)
    fig.tight_layout()
    fig.savefig(result_dir / 'correlations.png', dpi=140)
    plt.close(fig)


def plot_pairwise(summary, result_dir):
    available = summary[(summary['scope'] == 'all') & summary['pairwise_tasks'].gt(0)]
    if available.empty:
        (result_dir / 'pairwise_selection.png').unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = available.sort_values('pairwise_correct_fraction')
    labels = [f'{r.metric} (n={r.pairwise_tasks})' for r in available.itertuples()]
    fig, ax = plt.subplots(figsize=(7.5, max(3, .38 * len(labels))))
    ax.barh(labels, available['pairwise_correct_fraction'])
    ax.set(xlabel='Fraction selecting the better checkpoint', xlim=(0, 1),
           title='Two-checkpoint task comparisons (descriptive)')
    fig.tight_layout()
    fig.savefig(result_dir / 'pairwise_selection.png', dpi=140)
    plt.close(fig)


def target_support(y, task):
    y = np.asarray(y)
    if task == 'classification':
        values, counts = np.unique(y, return_counts=True)
        return json.dumps({str(int(value) + 1): int(count) for value, count in zip(values, counts)})
    if task == 'multi_label_classification':
        return json.dumps({str(i + 1): int(count) for i, count in enumerate(y.sum(axis=0))})
    return json.dumps({'min': float(y.min()), 'max': float(y.max())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE / 'config.yaml'))
    parser.add_argument('--modality', choices=['audio', 'video', 'audio-video'])
    parser.add_argument('--device')
    parser.add_argument('--samples-per-class', type=int)
    parser.add_argument('--samples-per-task', type=int, help='Total Probe-Train + Probe-Val cap; 0 disables it')
    parser.add_argument('--num-sentences', type=int)
    parser.add_argument('--num-audio-clips', type=int)
    parser.add_argument('--video-frames', type=int)
    parser.add_argument('--video-fps', type=int)
    parser.add_argument('--video-size', type=int)
    parser.add_argument('--decode-workers', type=int, help='Parallel video decoding threads (default: 1)')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--tasks', nargs='+', help='Explicit task IDs')
    parser.add_argument('--num-tasks', type=int, help='Use at most this many selected tasks')
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--checkpoints', nargs='+', help='Paths; checkpoint ID is each filename stem')
    parser.add_argument('--trajectory')
    parser.add_argument('--finetune-csv')
    parser.add_argument('--run-name', help='Write to results/<run-name> instead of results/')
    parser.add_argument('--metrics-only', action='store_true', help='Recompute from existing feature caches without GPU inference')
    args = parser.parse_args()
    with absolute(args.config).open(encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    for argument, key in [('modality', 'modality'), ('device', 'device'), ('samples_per_class', 'samples_per_class'),
                          ('samples_per_task', 'samples_per_task'),
                          ('num_sentences', 'num_sentences'), ('batch_size', 'batch_size'), ('seed', 'seed'),
                          ('num_audio_clips', 'num_audio_clips'), ('video_frames', 'video_frames'),
                          ('video_fps', 'video_fps'), ('video_size', 'video_size'), ('decode_workers', 'decode_workers'),
                          ('trajectory', 'trajectory'), ('finetune_csv', 'finetune_csv')]:
        value = getattr(args, argument)
        if value is not None:
            cfg[key] = value
    if args.checkpoints:
        cfg['checkpoints'] = [{'id': Path(path).stem, 'path': path} for path in args.checkpoints]
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('--run-name must contain only letters, numbers, _, -, or .')
    result_dir = HERE / 'results' / args.run_name if args.run_name else HERE / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    if len({x['id'] for x in cfg['checkpoints']}) != len(cfg['checkpoints']):
        raise ValueError('checkpoint IDs must be unique')
    for ckpt in cfg['checkpoints']:
        if not absolute(ckpt['path']).is_file():
            raise FileNotFoundError(f'checkpoint not found: {ckpt["path"]}')
    if int(cfg['batch_size']) < 1 or int(cfg['num_sentences']) < 1 or int(cfg['num_audio_clips']) < 1:
        raise ValueError('batch size, sentence count and audio clips must be positive')
    if int(cfg.get('decode_workers', 1)) < 1:
        raise ValueError('--decode-workers must be positive')
    if int(cfg['video_frames']) < 1 or int(cfg['video_fps']) < 1:
        raise ValueError('video frame count and fps must be positive')
    if args.num_tasks is not None and args.num_tasks < 1:
        raise ValueError('--num-tasks must be positive')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not args.metrics_only and not torch.cuda.is_available():
        raise RuntimeError(f'GPU device requested but CUDA is unavailable: {device}')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(HERE / 'logs/run.log', encoding='utf-8')])
    os.chdir(ROOT)
    torch.manual_seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']))
    catalog = read_csv(CATALOG)
    registry = {row['split_csv']: row for row in read_csv(REGISTRY)}
    names = args.tasks or ([r['task_id'] for r in catalog] if args.all_tasks else cfg['tasks'])
    if args.num_tasks:
        names = names[:args.num_tasks]
    if len(set(names)) != len(names):
        raise ValueError('selected task IDs must be unique')
    by_name = {r['task_id']: r for r in catalog}
    if set(names) - set(by_name):
        raise ValueError(f'unknown task IDs: {sorted(set(names) - set(by_name))}')
    # Snapshot the actual CLI selection, not just the defaults from YAML.
    cfg['tasks'] = list(names)
    prepared, statuses, audit, source_cache = [], [], [], {}
    protected = protected_subjects(catalog, cfg, registry, source_cache)
    abbreviation = {'audio': 'a', 'video': 'v', 'audio-video': 'va'}[cfg['modality']]
    for name in names:
        task = by_name[name]
        if abbreviation not in task['modalities'].split('|'):
            statuses.append({'task_id': name, 'checkpoint': '', 'status': 'skipped', 'reason': 'modality unavailable'})
            continue
        try:
            item = prepare_task(task, cfg, registry, source_cache, protected)
            prepared.append(item)
            audit.extend(item['audit'])
            LOG.info('Prepared %s: train=%d val=%d excluded-heldout-rows=%d unmapped-train-rows=%d',
                     name, item['n_train'], len(item['examples']) - item['n_train'],
                     item['excluded_heldout_rows'], item['missing_id'])
        except (ValueError, KeyError, OSError) as error:
            statuses.append({'task_id': name, 'checkpoint': '', 'status': 'skipped', 'reason': str(error)})
            LOG.warning('Skipped %s: %s', name, error)
    pd.DataFrame(audit, columns=['task_id', 'split', 'subject_hash', 'record_hash']).to_csv(result_dir / 'samples.csv', index=False)
    raw, pretrain_audit, manifest_cache = [], [], {}
    for ckpt in cfg['checkpoints']:
        if not prepared:
            break
        LOG.info('Loading checkpoint %s', ckpt['id'])
        try:
            if args.metrics_only:
                metadata = saved_checkpoint_metadata(ckpt)
                model = family = None
            else:
                model, family, metadata = build_model(absolute(ckpt['path']), cfg['modality'], cfg, device)
                saved_checkpoint_metadata(ckpt, metadata)
        except Exception as error:
            LOG.exception('Checkpoint %s failed to load', ckpt['id'])
            for item in prepared:
                statuses.append({'task_id': item['task']['task_id'], 'checkpoint': ckpt['id'], 'status': 'failed', 'reason': f'checkpoint: {type(error).__name__}: {error}'})
            continue
        try:
            audit_rows = audit_pretraining(prepared, ckpt, metadata, cfg['modality'], manifest_cache)
            pretrain_audit.extend(audit_rows)
            overlapping_tasks = sum(bool(row.get('exact_path_matches', 0) > 0) for row in audit_rows)
            if overlapping_tasks:
                LOG.warning('Current pretraining manifest contains held-out paths for %d task/modality rows; '
                            'historical checkpoint exposure is unverified', overlapping_tasks)
        except (OSError, ValueError, csv.Error) as error:
            LOG.warning('Pretraining manifest audit unavailable for %s: %s', ckpt['id'], error)
            pretrain_audit.extend({'task_id': p['task']['task_id'], 'checkpoint': ckpt['id'],
                'manifest': '', 'modality': cfg['modality'], 'status': 'audit_failed', 'heldout_paths': np.nan,
                'exact_path_matches': np.nan, 'manifest_newer_than_checkpoint': np.nan} for p in prepared)
        checkpoint_fields = {key: value for key, value in metadata.items() if key != 'pretrain_manifests'}
        for item in prepared:
            task = item['task']
            try:
                x, y = cached_features(item, ckpt, model, family, cfg, device)
                n = item['n_train']
                measures = evaluate(x[:n], y[:n], x[n:], y[n:], task['task_type'], int(cfg['regression_bins']), item['seed'])
                raw.append(dict(trajectory=cfg['trajectory'], modality=cfg['modality'], task_id=task['task_id'],
                    scope=task['scope'], task_type=task['task_type'], checkpoint=ckpt['id'],
                    n_train=n, n_val=len(y)-n, n_train_subjects=len({e['subject'] for e in item['examples'][:n]}),
                    n_val_subjects=len({e['subject'] for e in item['examples'][n:]}), feature_dim=x.shape[1],
                    train_target_support=target_support(y[:n], task['task_type']),
                    val_target_support=target_support(y[n:], task['task_type']),
                    sample_signature=item['signature'], **checkpoint_fields, **measures))
                statuses.append({'task_id': task['task_id'], 'checkpoint': ckpt['id'], 'status': 'ok', 'reason': ''})
                LOG.info('Evaluated %s / %s', ckpt['id'], task['task_id'])
            except Exception as error:
                # Exact-index decoding: never replace a failed item with another subject.
                detail = str(error)
                reason = type(error).__name__
                if detail and '/' not in detail and '\\' not in detail:
                    reason += ': ' + detail[:200]
                LOG.error('Failed %s / %s: %s', ckpt['id'], task['task_id'], reason)
                statuses.append({'task_id': task['task_id'], 'checkpoint': ckpt['id'], 'status': 'failed', 'reason': reason})
        del model
        gc.collect()
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(statuses, columns=['task_id', 'checkpoint', 'status', 'reason']).to_csv(result_dir / 'status.csv', index=False)
    pd.DataFrame(pretrain_audit, columns=['task_id', 'checkpoint', 'manifest', 'modality', 'status',
        'heldout_paths', 'exact_path_matches', 'manifest_newer_than_checkpoint']).to_csv(result_dir / 'pretrain_audit.csv', index=False)
    if not raw:
        raise RuntimeError(f'no task/checkpoint pair was evaluated; see {result_dir / "status.csv"}')
    raw = transfer_scores(pd.DataFrame(raw))
    raw.to_csv(result_dir / 'raw_metrics.csv', index=False)
    summary = summarize(raw)
    summary.to_csv(result_dir / 'transfer_summary.csv', index=False)
    plot_scores(summary, result_dir)
    ft = load_finetune(cfg)
    corr = correlations(raw, ft)
    corr.to_csv(result_dir / 'correlations.csv', index=False)
    correlation_summary = summarize_correlations(corr)
    correlation_summary.to_csv(result_dir / 'correlation_summary.csv', index=False)
    plot_correlations(correlation_summary, result_dir)
    plot_pairwise(correlation_summary, result_dir)
    fixed_checkpoint_baselines(raw, ft).to_csv(result_dir / 'checkpoint_baselines.csv', index=False)
    leave_one_task_out(corr).to_csv(result_dir / 'metric_selection.csv', index=False)
    leave_one_task_out(corr, selection='regret').to_csv(result_dir / 'metric_selection_regret.csv', index=False)
    with (result_dir / 'config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    LOG.info('Finished %d successful task/checkpoint pairs; %d correlation rows', len(raw), len(corr))
    if len(raw) != len(prepared) * len(cfg['checkpoints']) or len(prepared) != len(names):
        LOG.warning('Some requested tasks/checkpoints were skipped or failed; inspect %s', result_dir / 'status.csv')
    return 1 if any(row['status'] == 'failed' for row in statuses) else 0


if __name__ == '__main__':
    sys.exit(main())
