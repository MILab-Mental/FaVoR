"""VA task entries must retain the source tasks after merging shared presets."""
from pathlib import Path

import pytest
import yaml

from app.main import load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(path for group in ('classification', 'multilabel', 'regression', 'mental')
                 for path in (ROOT / f'CONFIGS/data/finetune/video/{group}').glob('*.yaml'))


@pytest.mark.parametrize('source', SOURCES, ids=lambda path: f'{path.parent.name}/{path.stem}')
def test_va_task_has_concrete_data_and_matching_targets(source):
    original = yaml.safe_load(source.read_text())['data']
    group, task = source.parent.name, source.stem
    config = load_config(ROOT / f'CONFIGS/tasks/finetune/audio-video/{group}/{task}.yaml')
    data = config['data']
    assert config['app'] == 'finetune_va'
    assert config['folder'] == f'OUTPUT/finetune_va/{group}/{task}'
    assert data['task'] == original['task']
    assert data['label_column'] == original['label_column']
    assert data['num_class'] == original.get('num_class', 1)
    expected = (['DATASET/merged/mental/clinical_canonical.csv'] if group == 'mental' else
                [f'DATASET/merged/finetune_va/{Path(path).name}' for path in original['datasets']])
    assert data['manifests'] == expected
    assert data['local_views'] == 0
    assert data['video_frames_global'] == data['video_fps'] * data['global_seconds']
    regression = original['task'] == 'regression'
    assert config['finetune']['best_metric'] == ('mae' if regression else 'f1_macro')
    assert config['finetune']['best_metric_mode'] == ('min' if regression else 'max')
    assert config['finetune']['pretrained_checkpoint']
    assert config['model']['fusion']['enabled']
    assert config['optimization']['epochs'] > 0


@pytest.mark.parametrize('group,task', [('classification', 'RAVDESS-emotion'),
                                     ('multilabel', 'MER242526-26openset'),
                                     ('regression', 'AVEC2014-PHQ')])
def test_removed_va_generic_paths_resolve_to_concrete_tasks(group, task):
    directory = ROOT / f'CONFIGS/tasks/finetune/audio-video/{group}'
    assert not (directory / 'default.yaml').exists()
    assert load_config(directory / 'default.yaml') == load_config(directory / f'{task}.yaml')
    data_directory = ROOT / f'CONFIGS/data/finetune/audio-video/{group}'
    assert not (data_directory / 'va.yaml').exists()
    assert load_config(data_directory / 'va.yaml') == load_config(data_directory / f'{task}.yaml')


def test_va_sampling_does_not_override_task_manifest():
    config = load_config(ROOT / 'CONFIGS/data/finetune/audio-video/sampling/va-6s.yaml')
    assert 'manifests' not in config['data']
