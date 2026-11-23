"""The catalog reproduces the archived standard task configurations."""
import csv
from pathlib import Path

import pytest

from app.main import apply_overrides, load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'CONFIGS/tasks/finetune'
with (BASE / 'tasks.csv').open(newline='', encoding='utf-8') as handle:
    TASKS = [row for row in csv.DictReader(handle) if row['enabled'] == '1']
CASES = [(task, modality) for task in TASKS for modality in task['modalities'].split('|')]


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
    assert current == old
