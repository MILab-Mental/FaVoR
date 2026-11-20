import csv
from pathlib import Path

import pytest
import yaml

from utils.finetune_catalog import read_tasks, task_data


ROOT = Path(__file__).resolve().parents[1]
TASKS = read_tasks()


@pytest.mark.parametrize('task', TASKS, ids=lambda row: row['task_id'])
def test_catalog_matches_archived_data(task):
    group = ('mental' if task['scope'] == 'mental' else
             {'classification': 'classification', 'regression': 'regression',
              'multi_label_classification': 'multilabel'}[task['task_type']])
    family = 'audio-video' if 'va' in task['modalities'].split('|') else 'audio'
    path = ROOT / 'CONFIGS/archive/finetune-data' / family / group / f'{task["task_id"]}.yaml'
    original = yaml.safe_load(path.read_text())['data']
    original.setdefault('datasets_weights', [1])
    original.setdefault('num_class', 1)
    assert task_data(task) == original


def test_legacy_paths_resolve_to_archived_data():
    from app.main import load_config

    for row in TASKS:
        group = ('mental' if row['scope'] == 'mental' else
                 {'classification': 'classification', 'regression': 'regression',
                  'multi_label_classification': 'multilabel'}[row['task_type']])
        family = 'audio-video' if 'va' in row['modalities'].split('|') else 'audio'
        path = ROOT / 'CONFIGS/data/finetune' / family / group / f'{row["task_id"]}.yaml'
        assert not path.exists()
        data = load_config(path)['data']
        data.setdefault('datasets_weights', [1])
        data.setdefault('num_class', 1)
        assert data == task_data(row)


def test_baseline_catalog_preserves_archived_tasks_and_order():
    from app.main import load_config
    from BASELINES.common.experiments import task_catalog

    archive = ROOT / 'CONFIGS/archive/finetune-data'
    expected = {}
    for family in ('audio', 'audio-video'):
        for group in ('classification', 'multilabel', 'regression'):
            for fragment in sorted((archive / family / group).glob('*.yaml')):
                expected['ordinary/' + fragment.stem] = load_config(fragment)['data']
    for fragment in sorted((archive / 'audio-video/mental').glob('*.yaml')):
        expected['mental/' + fragment.stem] = load_config(fragment)['data']
    actual = task_catalog()
    assert len(actual) == 71
    assert list(actual) == list(expected)
    assert actual == expected


def test_prepare_va_uses_catalog_without_generating_configs(tmp_path, monkeypatch):
    from utils import prepare_va_finetune as module

    entry = next(dict(task) for task in TASKS if task['task_id'] == 'CREMA-D-emotion')
    entry['csv_path'] = 'source.csv'
    (tmp_path / 'source.csv').write_text(
        'audio_path,video_path,emotion,emotion_split\n'
        'a.wav,a.mp4,1,0\nb.wav,b.mp4,2,1\n', encoding='utf-8')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'read_tasks', lambda root: [entry])
    monkeypatch.setattr(module, 'get_dataset_paths', lambda paths: [str(tmp_path)])
    module.prepare()
    with (tmp_path / 'DATASET/merged/finetune_va/tasks.csv').open(newline='') as handle:
        report = list(csv.DictReader(handle))
    assert report == [dict(group='classification', task='CREMA-D-emotion', train='1', validation='1')]
    with (tmp_path / 'DATASET/merged/finetune_va/source.csv').open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert not (tmp_path / 'CONFIGS').exists()


def test_prepare_clinical_uses_catalog_without_generating_configs(tmp_path, monkeypatch):
    from utils import prepare_clinical_mental as module

    tasks = [dict(row, num_class='2') for row in TASKS if row['scope'] == 'mental']
    names = [row['label_column'] for row in tasks]
    source = tmp_path / 'source.csv'
    fields = ['audio_path', 'video_path'] + [key for name in names for key in (name, name + '_split')]
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, label, split in [('a', '1', '0'), ('b', '2', '1')]:
            row = {'audio_path': name + '.wav', 'video_path': name + '.mp4'}
            row.update({key: value for task in names for key, value in ((task, label), (task + '_split', split))})
            writer.writerow(row)
    catalog = tmp_path / 'CONFIGS/datasets.csv'
    catalog.parent.mkdir(parents=True)
    catalog.write_text('split_csv,root_path,original_csv_path\n'
                       f'{module.SPLIT},old,old\n', encoding='utf-8')
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(module, 'read_tasks', lambda root: tasks)
    module.prepare(source)
    with (tmp_path / module.PAIRED).open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert (tmp_path / module.SPLIT).is_file()
    assert not (tmp_path / 'CONFIGS/tasks').exists()
    assert not (tmp_path / 'CONFIGS/data').exists()
