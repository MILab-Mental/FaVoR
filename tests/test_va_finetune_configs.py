"""VA task entries must retain the source tasks after merging shared presets."""
from pathlib import Path

import pytest
import yaml

from app.main import load_config
from utils.instance_finetune_configs import is_long_media_task


ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(path for group in ('classification', 'multilabel', 'regression', 'mental')
                 for path in (ROOT / f'CONFIGS/data/finetune/audio-video/{group}').glob('*.yaml'))


@pytest.mark.parametrize('source', SOURCES, ids=lambda path: f'{path.parent.name}/{path.stem}')
def test_va_task_has_concrete_data_and_matching_targets(source):
    original = yaml.safe_load(source.read_text())['data']
    group, task = source.parent.name, source.stem
    config = load_config(ROOT / f'CONFIGS/tasks/finetune/audio-video/{group}/{task}.yaml')
    data = config['data']
    assert config['app'] == 'finetune_va'
    suffix = '/instance-sentences8' if is_long_media_task(original) else ''
    assert config['folder'] == f'OUTPUT/finetune_va/{group}/{task}{suffix}'
    assert data['task'] == original['task']
    assert data['label_column'] == original['label_column']
    assert data['num_class'] == original.get('num_class', 1)
    assert data['datasets'] == original['datasets']
    assert 'manifests' not in data
    assert 'rootpaths' not in data
    assert not {'local_views', 'local_seconds', 'video_frames_local', 'video_local_size'} & data.keys()
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


def test_jepa_backbone_experiments_share_initialization_and_use_new_outputs():
    finetune = load_config(ROOT / 'CONFIGS/tasks/finetune/audio-video/mental/RESULT-total-jepa-backbones.yaml')
    pretrain = load_config(ROOT / 'CONFIGS/tasks/pretrain/audio-video/lejepa/jepa-e5-latest.yaml')
    assert finetune['app'] == 'finetune_va'
    assert pretrain['app'] == 'pretrain_va_lejepa'
    assert finetune['finetune']['pretrained_checkpoint'] is None
    assert finetune['data']['label_column'] == 'RESULT-total'
    assert finetune['folder'].endswith('instance-sentences8-jepa-backbones')
    assert pretrain['folder'].endswith('JEPA-e5-latest-C1')
    for cfg in (finetune, pretrain):
        assert cfg['meta']['read_checkpoint'] is None
        assert cfg['model']['video']['init_source'] == 'vjepa'
        assert cfg['model']['audio']['init_source'] == 'audio_jepa'
        assert cfg['model']['fusion']['enabled'] == ['feature']
        assert cfg['model']['video']['vjepa_checkpoint'].endswith('FaVoR-112px-48f/e5.pt')
        assert cfg['model']['audio']['audio_jepa_checkpoint'].endswith('FaVoR-2to6s/latest.pt')
    assert finetune['model']['video']['vjepa_checkpoint'] == pretrain['model']['video']['vjepa_checkpoint']
    assert finetune['model']['audio']['audio_jepa_checkpoint'] == pretrain['model']['audio']['audio_jepa_checkpoint']


def test_cli_overrides_clear_checkpoints_and_keep_empty_sync_path():
    from app.main import apply_overrides

    config = {'meta': {'read_checkpoint': 'old.pt'},
              'finetune': {'pretrained_checkpoint': 'old_va.pt'},
              'data': {'sync_preflight_file': 'shared.json'}}
    apply_overrides(config, ['meta.read_checkpoint=null', 'finetune.pretrained_checkpoint=~',
                            'data.sync_preflight_file='])
    assert config['meta']['read_checkpoint'] is None
    assert config['finetune']['pretrained_checkpoint'] is None
    assert config['data']['sync_preflight_file'] == ''
