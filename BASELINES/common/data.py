"""Use FAVOR manifests/labels/identities, decoding raw media before native transforms.

No failed sample substitution, no missing-modality padding, and no test-set selection.
"""
import csv
import hashlib
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.audio_jepa.finetune_dataset import AudioCSVDataset
from datasets.audio_jepa.pretrain_dataset import load_audio
from datasets.common.instances import sentence_identity, sentence_windows, sample_sentence_indices
from datasets.common.paths import get_dataset_paths
from datasets.va_lejepa.av_decoder import AVDecoder
from BASELINES.common.runtime import path, fingerprint


def configure_instances(cfg, modality):
    """Use FAVOR's long-media sampling and instance-head presets verbatim."""
    from app.main import load_config
    from utils.instance_finetune_configs import is_long_media_task, INSTANCE_PRESETS
    data = cfg['data']
    if not is_long_media_task(data) and not data.get('num_sentence', 0):
        return
    app = {'audio':'finetune_a', 'video':'finetune_v', 'audio-video':'finetune_va'}[modality]
    family, sampling, model = INSTANCE_PRESETS[app]
    sampling_path = f'CONFIGS/data/finetune/{family}/sampling/{sampling}'
    model_path = f'CONFIGS/models/finetune/{family}/{model}'
    native = load_config(path(sampling_path))['data']
    if not data.get('num_sentence'):
        data['num_sentence'] = native['num_sentence']
    data.setdefault('sentence_eval_stride', native['sentence_eval_stride'])
    cfg.setdefault('model', {}).setdefault('instance_aggregator',
        load_config(path(model_path))['model']['instance_aggregator'])
    cfg['instance_protocol'] = dict(identity='FAVOR sentence_identity',
        sampling_preset=sampling_path, head_preset=model_path,
        prediction_unit='speaker_recording', csv_splits_unchanged=True)


def read_rows(cfg, modality, supervised=True):
    manifests = cfg['datasets']
    roots = cfg.get('rootpaths') or get_dataset_paths(manifests)
    if len(roots) != len(manifests):
        raise ValueError('datasets/rootpaths length mismatch')
    records, missing = [], 0
    for manifest, root in zip(manifests, roots):
        with path(manifest).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if supervised:
                    split = row.get(cfg['label_column'] + '_split', '')
                    if split in ('', '-1'):
                        continue
                    row['_split'] = int(split)
                if cfg.get('video_column'):
                    if cfg['video_column'] not in row:
                        raise ValueError(f'{manifest}: missing {cfg["video_column"]}; a lip model cannot consume full-face video')
                    row['video_path'] = row[cfg['video_column']]
                # AV comparisons can enforce the same paired cohort for every branch.
                required = ('audio', 'video') if cfg.get('cohort', 'paired') == 'paired' or modality == 'audio-video' else (modality,)
                if any(not row.get(m + '_path', '').strip() for m in required):
                    missing += 1
                    continue
                for field in ('audio_path', 'video_path'):
                    if row.get(field):
                        value = Path(row[field])
                        if '|' in row[field]:
                            raise ValueError('Multi-path row needs an explicit segment manifest')
                        row[field] = str(value if value.is_absolute() else path(root) / value)
                row['_manifest'] = str(manifest)
                if supervised:
                    # Exactly FAVOR's one-based class, pipe-delimited multi-label contract.
                    parser = object.__new__(AudioCSVDataset)
                    parser.task, parser.num_class = cfg['task'], int(cfg.get('num_class', 1))
                    label = parser._parse_label(row[cfg['label_column']], manifest, len(records) + 2)
                    if parser.task == 'classification' and not 0 <= label < parser.num_class:
                        raise ValueError(f'Label outside [1, num_class] in {manifest}')
                    if not np.isfinite(label).all():
                        raise ValueError(f'Nonfinite label in {manifest}')
                    row['_label'] = label
                records.append(row)
    return records, missing


def partition(rows, cfg=None):
    """Use CSV assignments verbatim: 0=train, 1=test, optional 2=validation."""
    names = {0: 'train', 1: 'test', 2: 'validation'}
    splits = {name: [] for name in names.values()}
    for row in rows:
        if row['_split'] not in names:
            raise ValueError('Supported task CSV splits: 0=train, 1=test, 2=validation')
        splits[names[row['_split']]].append(row)
    if not splits['train']:
        raise ValueError('The CSV contains no training samples for this task')
    return {name: values for name, values in splits.items() if values}


class MediaDataset(Dataset):
    def __init__(self, rows, cfg, spec, training, supervised=True):
        self.cfg, self.spec, self.training, self.supervised = cfg, spec, training, supervised
        count, stride = int(cfg.get('num_sentence', 0)), int(cfg.get('sentence_eval_stride', 1))
        if count < 0 or (stride != -1 and stride < 1):
            raise ValueError('num_sentence must be nonnegative; sentence_eval_stride must be -1 or positive')
        groups = OrderedDict()
        modality = spec['modality']
        for row in rows:
            filename = row.get('video_path') or row['audio_path']
            identity = sentence_identity(filename)[0] if cfg.get('num_sentence', 0) else filename
            groups.setdefault(identity, []).append(row)
        self.samples = list(groups)
        self.rows = [sorted(values, key=lambda r: sentence_identity(r.get('video_path') or r['audio_path'])[1]) for values in groups.values()]
        self.labels = [values[0]['_label'] for values in self.rows] if supervised else []
        for values in self.rows:
            if supervised and any(not np.array_equal(values[0]['_label'], r['_label']) for r in values):
                raise ValueError('Inconsistent labels across sentences of one media instance')
        self.modality = modality

    def __len__(self):
        return len(self.rows)

    def _decode(self, row):
        spec = self.spec
        seconds = float(spec['seconds'])
        sample_rate = int(spec.get('sample_rate', 16000))
        has_audio = self.modality in ('audio', 'audio-video')
        has_video = self.modality in ('video', 'audio-video')
        if self.modality == 'audio-video':
            # FAVOR verifies source identities, timestamps, and common valid duration.
            from datasets.va_lejepa.va_manifest import source_key, validate_identity
            pair = dict(row)
            pair.setdefault('source_id', source_key(row['video_path']))
            pair.setdefault('pair_id', hashlib.sha256(row['video_path'].encode()).hexdigest()[:24])
            validate_identity(pair)
            decoder = AVDecoder(pair, sample_rate, self.cfg.get('sync', {}))
            duration = decoder.video_duration
            if spec.get('favor_data_cfg'):
                from datasets.va_lejepa.dataset import VALeJEPADataset
                from datasets.va_lejepa.av_time_sampler import AVTimeSampler
                from datasets.va_lejepa.transforms import AVTransforms
                native = object.__new__(VALeJEPADataset)
                native.cfg = spec['favor_data_cfg']
                native.transforms = AVTransforms(native.cfg, self.training)
                sampler = AVTimeSampler(global_seconds=seconds, local_views=0, training=self.training,
                    short_policy=native.cfg.get('short_policy', 'zero_pad'))
                interval, _ = sampler(duration, duration)
                view = native._decode_view(decoder, interval)
                view['_favor_prepared'] = True
                return [view]
        else:
            if has_audio:
                waveform = load_audio(row['audio_path'], sample_rate)
                duration = len(waveform) / sample_rate
            if has_video:
                from decord import VideoReader, cpu
                reader = VideoReader(row['video_path'], ctx=cpu(0), num_threads=1)
                fps = float(reader.get_avg_fps())
                duration = len(reader) / fps
                if spec.get('frame_stride'):
                    seconds = int(spec['frames']) * int(spec['frame_stride']) / fps
        if duration <= 0:
            raise ValueError('Empty media: ' + (row.get('audio_path') or row['video_path']))
        span = min(seconds, duration)
        maximum = max(0., duration - span)
        count = int(spec.get('pretrain_clips', 1)) if not self.supervised else 1
        starts = [random.uniform(0, maximum) for _ in range(count)] if self.training else np.linspace(0, maximum, max(count, int(self.cfg.get('eval_clips', 1)))).tolist()
        clips = []
        for start in starts:
            if self.modality == 'audio-video':
                frames = min(int(spec['frames']), max(1, round(span*spec['fps']))) if spec.get('fps') else int(spec['frames'])
                frames = min(frames, max(1, int(span*decoder.fps)))
                clip = decoder.decode((start, start + span), frames)
                clip['video_valid_frames'] = torch.tensor(frames)
            else:
                clip = {}
                if has_audio:
                    audio = waveform[round(start * sample_rate):round((start + span) * sample_rate)]
                    clip.update(audio=audio, audio_lengths=torch.tensor(len(audio)))
                if has_video:
                    indices = np.minimum(len(reader)-1, np.floor(np.linspace(start, start + span, int(spec['frames']), endpoint=False) * fps).astype('int64'))
                    clip['video'] = torch.from_numpy(reader.get_batch(indices).asnumpy())
                    clip.update(video_lengths=torch.tensor(len(indices)), source_fps=torch.tensor(fps),
                                start_time=torch.tensor(start), end_time=torch.tensor(start+span))
            clips.append(clip)
        return clips

    def __getitem__(self, index):
        rows = self.rows[index]
        count = int(self.cfg.get('num_sentence', 0))
        indices = list(range(len(rows)))
        if self.training:
            selected = sample_sentence_indices(indices, count) if count else indices
            windows = [list(range(len(selected)))]
        else:
            stride = int(self.cfg.get('sentence_eval_stride', 1))
            if count and stride == -1:
                selected = sample_sentence_indices(indices, count)
                windows = [list(range(len(selected)))]
            else:
                selected = indices
                windows = sentence_windows(indices, count, stride) if count else [indices]
        decoded = {i: self._decode(rows[i]) for i in set(selected)}
        return dict(sentences=[decoded[i] for i in selected], windows=windows,
                    label=self.labels[index] if self.supervised else None, id=self.samples[index])


class PretrainDataset(MediaDataset):
    """Reuse FAVOR's mmap line index instead of materializing millions of rows."""
    def __init__(self, cfg, spec):
        from datasets.audio_jepa.pretrain_dataset import AudioPretrainDataset
        self.cfg, self.spec = cfg, spec
        self.training, self.supervised, self.modality = True, False, spec['modality']
        self.records = AudioPretrainDataset([path(p) for p in cfg['datasets']])
        self.canonical = self.modality == 'audio-video'
        if self.canonical and len(cfg['datasets']) != 1:
            raise ValueError('Use the single canonical audio-video pretraining CSV')
        if self.canonical:
            with path(cfg['datasets'][0]).open(encoding='utf-8-sig', newline='') as handle:
                self.columns = next(csv.reader(handle))
            if not {'pair_id', 'source_id', 'audio_path', 'video_path'}.issubset(self.columns):
                raise ValueError('Expected the canonical paired pretraining CSV')
        self.full_size = len(self.records) - int(self.canonical)
        self.size = min(self.full_size, int(cfg.get('limit') or self.full_size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        value = self.records._audio_path(index + int(self.canonical))
        if self.canonical:
            from datasets.va_lejepa.va_manifest import validate_identity
            row = dict(zip(self.columns, next(csv.reader([value]))))
            validate_identity(row)
        else:
            row = {self.modality + '_path': value}
        clips = self._decode(row)
        return dict(sentences=[clips], windows=[[0]], label=None,
                    id=row.get('video_path') or row['audio_path'])


def make_loaders(cfg, spec, pretrain=False):
    data = cfg['data']
    if not pretrain and data['task'] == 'regression':
        data['num_class'] = 1
    if not pretrain:
        configure_instances(cfg, spec['modality'])
    if spec.get('requires_lips') and data.get('video_column') != 'lip_video_path':
        raise ValueError('AV-HuBERT requires lip_video_path input')
    if pretrain:
        datasets = {'train': PretrainDataset(data, spec)}
        missing, roots = 0, []
    else:
        rows, missing = read_rows(data, spec['modality'])
        datasets = {name: MediaDataset(values, data, spec, name=='train')
                    for name, values in partition(rows).items()}
        roots = [str(path(p).resolve()) for p in (data.get('rootpaths') or get_dataset_paths(data['datasets']))]
    audit = dict(manifests={str(p):fingerprint(p) for p in data['datasets']}, media_roots=roots,
                 missing_modality_rows=missing, split_policy='csv_only', splits={})
    loaders = {}
    for name, dataset in datasets.items():
        training = name == 'train'
        count = dataset.full_size if pretrain else sum(len(rows) for rows in dataset.rows)
        audit['splits'][name] = dict(rows=count, instances=dataset.full_size if pretrain else len(dataset))
        limit = int(data.get('limit', 0))
        if limit and not pretrain:
            dataset.rows, dataset.samples = dataset.rows[:limit], dataset.samples[:limit]
            dataset.labels = dataset.labels[:limit]
        generator = torch.Generator().manual_seed(int(cfg.get('seed', 42)))
        batch_size = int(data.get('batch_size', 1))
        if batch_size < int(spec.get('minimum_batch_size', 1)):
            raise ValueError(f'Native objective requires batch_size >= {spec["minimum_batch_size"]}')
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=training,
            num_workers=int(data.get('num_workers', 0)), collate_fn=list, generator=generator,
            persistent_workers=False, drop_last=training and spec.get('minimum_batch_size', 1)>1)
    cfg['data_audit'] = audit
    audit['prediction_unit'] = cfg.get('instance_protocol', {}).get('prediction_unit', 'clip') if not pretrain else 'pretraining_clip'
    return loaders
