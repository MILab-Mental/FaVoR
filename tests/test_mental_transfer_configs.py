"""Mental-auto entries retain tuned parameters and each task definition."""
from pathlib import Path
import csv
import os
import shlex
import subprocess

import pytest
import yaml

from app.main import load_config
from app.main import apply_overrides

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / 'CONFIGS/archive/finetune-mental-auto'
RESULTS = {'total-best': 'RESULT-total', 'anxiety-best': 'RESULT-Anxiety',
           'depression-best': 'RESULT-Depression'}
SOURCES = {
    'total-best': 'recover-total-best/total.yaml',
    'anxiety-best': 'lr31e5-Anxiety/Anxiety.yaml',
    'depression-best': 'init75k-Depression/Depression.yaml',
}
TASKS = sorted(p for p in (ROOT / 'CONFIGS/archive/finetune-data/audio-video/mental').glob('*.yaml')
               if not p.stem.startswith('RESULT-'))


@pytest.mark.parametrize('group', SOURCES)
def test_transfer_preset_matches_tuned_source(group):
    config = load_config(ENTRIES / f'{RESULTS[group]}.yaml')
    assert config['folder'] == f'OUTPUT/finetune_va/mental/{RESULTS[group]}/autoresearch-1026'
    raw = yaml.safe_load((ENTRIES / f'{RESULTS[group]}.yaml').read_text())
    assert 'model' not in raw
    assert 'epochs' not in raw['optimization']
    assert raw['yamls']['opt'] == 'CONFIGS/optimization/finetune/audio-video/epochs100-warmup0-lr0.001.yaml'
    assert len(TASKS) == 35
    source_path = ROOT / 'autoresearch/oct15' / SOURCES[group]
    if not source_path.is_file():
        pytest.skip('local autoresearch provenance artifact is not available')
    source = yaml.safe_load(source_path.read_text())
    assert config['model'] == source['model']
    assert config['optimization'] == dict(source['optimization'], epochs=100, early_stopping_patience=10)
    expected_finetune = dict(source['finetune'])
    expected_finetune.pop('early_stopping_patience', None)
    assert config['finetune'] == expected_finetune
    for key, value in source['data'].items():
        if key != 'sync_preflight_file':
            assert config['data'][key] == value


@pytest.mark.parametrize('group', SOURCES)
@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.stem)
def test_transfer_entry_preserves_task(group, task):
    config = load_config(ENTRIES / f'{task.stem}-auto-{group}.yaml')
    reference = load_config(ENTRIES / f'{RESULTS[group]}.yaml')
    for key in ['model', 'optimization', 'finetune', 'meta']:
        assert config[key] == reference[key]
    expected = yaml.safe_load(task.read_text())['data']
    folder = f'OUTPUT/finetune_va/mental-transfer/{task.stem}/{group}'
    for key, value in expected.items():
        assert config['data'][key] == value
    assert config['app'] == 'finetune_va'
    assert config['folder'] == folder
    assert config['optimization']['epochs'] == 100
    assert config['optimization']['early_stopping_patience'] == 10
    assert 'early_stopping_patience' not in config['finetune']
    assert config['meta']['auto_resume'] is True
    assert 'autoresearch/' not in config['data']['sync_preflight_file']


def test_mental_auto_layout():
    expected = {f'{result}.yaml' for result in RESULTS.values()}
    expected.update(f'{task.stem}-auto-{group}.yaml' for task in TASKS for group in SOURCES)
    assert {p.name for p in ENTRIES.glob('*.yaml')} == expected
    assert len(expected) == 108
    assert all(not (ENTRIES.parent / f'mental-{group}.yaml').exists() for group in SOURCES)


@pytest.mark.parametrize('route', SOURCES)
def test_launcher_reproduces_transferred_configs(route, tmp_path):
    source = ROOT / 'CONFIGS/tasks/finetune/tasks.csv'
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    dataset = tmp_path / 'dataset.csv'
    dataset.touch()
    catalog = tmp_path / 'tasks.csv'
    with catalog.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row, csv_path=str(dataset)) for row in rows)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.touch()
    env = {**os.environ, 'MENTAL_AUTO': route, 'SCOPE': 'mental', 'DRY_RUN': '1',
           'CUDA_VISIBLE_DEVICES': '0', 'CHECKPOINT': str(checkpoint),
           'TASK_CSV': str(catalog), 'TASKS': '', 'TYPE': 'all'}
    result = subprocess.run(['bash', str(ROOT / 'train_finetune.sh'), 'va',
                             f'folder={tmp_path / "preview"}'], cwd=ROOT, env=env,
                            text=True, capture_output=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('torchrun ')]
    assert len(commands) == 38
    by_label = {}
    for command in commands:
        options = command[command.index('--set') + 1:]
        label = next(item.split('=', 1)[1] for item in options if item.startswith('data.label_column='))
        by_label[label] = (command, options)
    assert len(by_label) == 38
    legacy = [(task.stem, ENTRIES / f'{task.stem}-auto-{route}.yaml') for task in TASKS]
    legacy.append((RESULTS[route], ENTRIES / f'{RESULTS[route]}.yaml'))
    for label, old_path in legacy:
        command, options = by_label[label]
        current = load_config(ROOT / command[command.index('--fname') + 1])
        apply_overrides(current, options)
        original = load_config(old_path)
        route_config = load_config(ROOT / f'CONFIGS/tasks/finetune/audio-video/mental-auto/{route}.yaml')
        assert route_config['finetune']['pretrained_checkpoint'] == original['finetune']['pretrained_checkpoint']
        original['folder'] = str(tmp_path / 'preview')
        original['data']['datasets'] = [str(dataset)]
        original['finetune']['pretrained_checkpoint'] = str(checkpoint)
        assert current == original


def test_mental_auto_only_has_three_shared_routes():
    entries = ROOT / 'CONFIGS/tasks/finetune/audio-video/mental-auto'
    assert {p.name for p in entries.glob('*.yaml')} == {f'{route}.yaml' for route in SOURCES}
