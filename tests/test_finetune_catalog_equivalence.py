"""Compare catalog defaults with the archived tasks and the current sampling policy."""
import csv
from pathlib import Path

import pytest

from app.main import apply_overrides, load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'CONFIGS/tasks/finetune'
with (BASE / 'tasks.csv').open(newline='', encoding='utf-8') as handle:
    TASKS = [row for row in csv.DictReader(handle) if row['enabled'] == '1']
CASES = [(task, modality) for task in TASKS for modality in task['modalities'].split('|')]


def expect_clip_sampling(config, modality):
    """Apply the clip profile to an ordinary task archived with eight sentences."""
    data = config['data']
    assert data['num_sentence'] == 8
    del data['num_sentence'], data['sentence_eval_stride'], config['model']['instance_aggregator']

    if modality == 'a':
        data['num_clips'] = 4
    elif modality == 'v':
        data.update(num_workers=8, persistent_workers=True)
        del data['val_batch_size'], data['val_num_workers'], config['meta']['frozen_encoder']
    else:
        del data['group_by_lengths'], data['short_video_full_frames'], data['sync_preflight_file']
        del data['sync']['duration_tolerance_frames']
        data['sync'].update(duration_tolerance_seconds=0.02, offset_tolerance_seconds=0.0)


@pytest.mark.parametrize('task,modality', CASES,
                         ids=[f'{task["task_id"]}-{mode}' for task, mode in CASES])
def test_catalog_matches_archived_task(task, modality):
    group = ('mental' if task['scope'] == 'mental' else
             {'classification': 'classification', 'regression': 'regression',
              'multi_label_classification': 'multilabel'}[task['task_type']])
    family = {'a': 'audio', 'v': 'video', 'va': 'audio-video'}[modality]
    old_path = ROOT / 'CONFIGS/archive/finetune-standard' / family / group / f'{task["task_id"]}.yaml'
    old = load_config(old_path)
    current = load_config(BASE / 'defaults' / modality / task['profile'] / 'default.yaml')
    overrides = [f'data.datasets=["{task["csv_path"]}"]', 'data.datasets_weights=[1]',
                 f'data.task={task["task_type"]}', f'data.num_class={task["num_class"]}',
                 f'data.label_column={task["label_column"]}', f'folder={old["folder"]}']
    if modality == 'va':
        overrides += [f'finetune.best_metric={task["best_metric"]}',
                      f'finetune.best_metric_mode={task["best_metric_mode"]}']
    if task['overrides']:
        overrides += task['overrides'].split(';')
    if modality == 'va':
        overrides.append(f'finetune.pretrained_checkpoint={old["finetune"]["pretrained_checkpoint"]}')
        # Both absent and null disable full-state resume.
        current.setdefault('meta', {})['read_checkpoint'] = None
    else:
        overrides.append(f'meta.read_checkpoint={old["meta"]["read_checkpoint"]}')
    apply_overrides(current, overrides)
    # The launcher makes these existing trainer defaults explicit.
    old.setdefault('meta', {}).setdefault('save_latest', False)
    if modality == 'a':
        old['data'].setdefault('datasets_weights', [1])
        if task['task_type'] == 'regression':
            old['data'].setdefault('num_class', 1)
    if modality in ('a', 'v') and task['profile'] == 'clip':
        assert current['model'].pop('regression_head') == 'layer_weighted'
    if task['scope'] == 'nonmental' and old['data'].get('num_sentence', 0):
        assert task['profile'] == 'clip'
        expect_clip_sampling(old, modality)
    assert current == old
