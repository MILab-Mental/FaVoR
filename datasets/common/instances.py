"""Sentence sampling and instance batches shared by downstream modalities."""
from collections import OrderedDict
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


def sentence_identity(path):
    """Keep the media directory in the identity; order -sXXXX numerically."""
    path = Path(path)
    match = re.fullmatch(r"(.*)-s(\d+)", path.stem)
    return str(path.parent / (match[1] if match else path.stem)), int(match[2]) if match else 0


def sentence_windows(indices, count, stride=1):
    if count < 1 or stride < 1:
        raise ValueError("sentence count and evaluation stride must be positive")
    if not indices:
        raise ValueError("an instance must contain at least one sentence")
    if len(indices) <= count:
        return [list(indices) + [indices[-1]] * (count - len(indices))]
    starts = list(range(0, len(indices) - count + 1, stride))
    if starts[-1] != len(indices) - count:
        starts.append(len(indices) - count)
    return [list(indices[start:start + count]) for start in starts]


def sample_sentence_indices(group, count):
    selected = sorted(random.sample(group, count)) if len(group) > count else list(group)
    return selected + [selected[-1]] * (count - len(selected))


def combine_sentences(sentences, modality, instance_id):
    """Add an explicit sentence axis, preserving each modality's inner views."""
    if modality == "audio-video":
        errors = [s['decode_error'] for s in sentences if 'decode_error' in s]
        if errors:
            return {'decode_error': dict(errors[0], instance_id=instance_id)}
        first = sentences[0]
        return {
            'global': default_collate([s['global'] for s in sentences]),
            'local': [], 'label': first['label'],
            'pair_id': instance_id, 'source_id': instance_id,
            'video_path': sentence_identity(first['video_path'])[0],
            'audio_path': sentence_identity(first['audio_path'])[0],
            'sync_diagnostics': [s['sync_diagnostics'] for s in sentences],
        }
    first = sentences[0]
    inputs = default_collate([s[0] for s in sentences])
    if modality == "audio":
        return inputs, first[1], instance_id
    return inputs, first[1], default_collate([s[2] for s in sentences]), instance_id


def collate_instance_va(samples):
    skipped = [s['decode_error'] for s in samples if 'decode_error' in s]
    valid = [s for s in samples if 'decode_error' not in s]
    if not valid:
        return {'skipped_samples': skipped}
    batch = {'global': default_collate([s['global'] for s in valid]),
             'local': {'video': torch.empty(len(valid), 0)},
             'label': torch.stack([s['label'] for s in valid]), 'skipped_samples': skipped}
    for key in ('pair_id', 'source_id', 'video_path', 'audio_path', 'sync_diagnostics'):
        batch[key] = [s[key] for s in valid]
    return batch


def collate_instance_evaluation(samples):
    skipped = [s['decode_error'] for s in samples if 'decode_error' in s]
    return {'instance_samples': [s for s in samples if 'decode_error' not in s],
            'skipped_samples': skipped}


class SentenceInstanceDataset(Dataset):
    """One dataset item and label per original media instance.

    Evaluation decodes each sentence once, retains window indices, and stacks
    only the window about to be predicted instead of copying every overlap.
    """
    def __init__(self, dataset, num_sentence, modality, training, eval_stride=1):
        self.dataset, self.modality, self.training = dataset, modality, bool(training)
        self.num_sentence, self.eval_stride = int(num_sentence), int(eval_stride)
        if self.num_sentence < 1 or (self.eval_stride != -1 and self.eval_stride < 1):
            raise ValueError('num_sentence must be positive; sentence_eval_stride must be -1 or positive')
        paths = ([row['video_path'] for row in dataset.rows]
                 if modality == 'audio-video' else dataset.samples)
        groups = OrderedDict()
        for index, path in enumerate(paths):
            identity, order = sentence_identity(path)
            groups.setdefault(identity, []).append((order, index))
        self.samples = list(groups)
        self.groups = [[index for _, index in sorted(group)] for group in groups.values()]
        self.labels = [dataset.labels[group[0]] for group in self.groups]
        for identity, group, label in zip(self.samples, self.groups, self.labels):
            if any(not np.array_equal(dataset.labels[index], label) for index in group):
                raise ValueError(f'inconsistent sentence labels for instance {identity}')

    @property
    def epoch(self):
        return getattr(self.dataset, 'epoch', 0)

    @epoch.setter
    def epoch(self, value):
        self.dataset.epoch = value

    @property
    def collate_fn(self):
        if not self.training:
            return collate_instance_evaluation
        return collate_instance_va if self.modality == 'audio-video' else default_collate

    def __len__(self):
        return len(self.groups)

    def _sentence(self, index):
        # Exact-index decoding must never replace a failed sentence with a
        # sentence belonging to a different instance and label.
        return self.dataset.get_sentence(index)

    def __getitem__(self, index):
        group, identity = self.groups[index], self.samples[index]
        if self.training:
            selected = sample_sentence_indices(group, self.num_sentence)
            decoded = {i: self._sentence(i) for i in set(selected)}
            return combine_sentences([decoded[i] for i in selected], self.modality, identity)
        if self.eval_stride == -1:
            selected = sample_sentence_indices(group, self.num_sentence)
            decoded = {i: self._sentence(i) for i in set(selected)}
            sentences = [decoded[i] for i in selected]
            windows = [list(range(self.num_sentence))]
        else:
            sentences = [self._sentence(i) for i in group]
            windows = sentence_windows(list(range(len(group))), self.num_sentence, self.eval_stride)
        if self.modality == 'audio-video':
            errors = [s['decode_error'] for s in sentences if 'decode_error' in s]
            if errors:
                return {'decode_error': dict(errors[0], instance_id=identity)}
        return {'sentences': sentences,
                'windows': windows,
                'label': torch.as_tensor(self.labels[index]), 'instance_id': identity, 'modality': self.modality}


def as_sentence_instances(dataset, cfg, modality, training):
    count = int(cfg.get('num_sentence', 0))
    if count < 0:
        raise ValueError('data.num_sentence must be >= 0')
    return (SentenceInstanceDataset(dataset, count, modality, training,
                                   cfg.get('sentence_eval_stride', 1)) if count else dataset)


def instance_window_batches(sample):
    collate = collate_instance_va if sample['modality'] == 'audio-video' else default_collate
    for indices in sample['windows']:
        combined = combine_sentences([sample['sentences'][i] for i in indices],
                                     sample['modality'], sample['instance_id'])
        yield collate([combined])
