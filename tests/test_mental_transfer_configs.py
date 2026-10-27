"""Mental-auto entries retain tuned parameters and each task definition."""
from pathlib import Path

import pytest
import yaml

from app.main import load_config

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / 'CONFIGS/tasks/finetune/audio-video/mental-auto'
RESULTS = {'total-best': 'RESULT-total', 'anxiety-best': 'RESULT-Anxiety',
           'depression-best': 'RESULT-Depression'}
SOURCES = {
    'total-best': 'recover-total-best/total.yaml',
    'anxiety-best': 'lr31e5-Anxiety/Anxiety.yaml',
    'depression-best': 'init75k-Depression/Depression.yaml',
}
TASKS = sorted(p for p in (ROOT / 'CONFIGS/data/finetune/audio-video/mental').glob('*.yaml')
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
