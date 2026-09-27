"""Every modality uses shared task data while retaining its own training preset."""
from pathlib import Path

import pytest
import yaml

from app.main import load_config

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('classification', 'regression', 'multilabel', 'mental')
SHARED = sorted(path for group in GROUPS
                for path in (ROOT / 'CONFIGS/data/finetune/audio-video' / group).glob('*.yaml'))


@pytest.mark.parametrize('fragment', SHARED, ids=lambda path: f'{path.parent.name}/{path.stem}')
def test_three_modalities_share_task_data(fragment):
    expected = yaml.safe_load(fragment.read_text())['data']
    for modality, app in [('audio', 'finetune_a'), ('video', 'finetune_v'), ('audio-video', 'finetune_va')]:
        task = ROOT / 'CONFIGS/tasks/finetune' / modality / fragment.parent.name / fragment.name
        raw = yaml.safe_load(task.read_text())
        assert raw['yamls']['data'] == str(fragment.relative_to(ROOT))
        assert f'/finetune/{modality}/sampling/' in raw['yamls']['sampling']
        merged = load_config(task)
        assert merged['app'] == app
        for key, value in expected.items():
            assert merged['data'][key] == value


def test_only_audio_exclusive_task_fragments_remain():
    base = ROOT / 'CONFIGS/data/finetune'
    for group in GROUPS:
        assert not (base / 'video' / group).exists()
        for fragment in (base / 'audio' / group).glob('*.yaml'):
            assert not (base / 'audio-video' / group / fragment.name).exists()
            data = yaml.safe_load(fragment.read_text())['data']
            assert all(Path(path).stem.endswith(('SA', 'LA')) for path in data['datasets'])
    assert len(SHARED) == 53
    assert sum(len(list((base / 'audio' / group).glob('*.yaml'))) for group in GROUPS) == 18
