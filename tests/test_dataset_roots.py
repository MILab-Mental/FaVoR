"""Shared roots apply to all modalities and invalidate paired path caches."""
from pathlib import Path

import pytest

from datasets.common import paths
from datasets.va_lejepa.dataset import VALeJEPADataset
from datasets.audio_jepa.finetune_dataset import AudioCSVDataset


def test_shared_registry_and_cache_refresh(tmp_path, monkeypatch):
    split = tmp_path / 'pairs.csv'
    split.write_text('audio_path,video_path,emotion,emotion_split\na/x.wav,v/x.mp4,1,0\na/y.wav,v/y.mp4,2,1\n')
    registry = tmp_path / 'datasets.csv'
    monkeypatch.setattr(paths, 'DATASETS_CSV', registry)
    def move_root(name):
        media = tmp_path / name
        registry.write_text(f'split_csv,root_path,original_csv_path\n{split},{media},unused\n')
        return media
    media = move_root('first')
    assert paths.get_dataset_paths([str(split)]) == [str(media)]
    audio = AudioCSVDataset([str(split)], 0, label_column='emotion', task='classification')
    assert audio.samples == [str(media / 'a/x.wav')]
    from datasets.video_jepa.finetune_dataset import VideoCSVDataset
    video = VideoCSVDataset([str(split)], 0, label_column='emotion', task='classification')
    assert video.samples == [str(media / 'v/x.mp4')]
    first = VALeJEPADataset(dict(datasets=[str(split)]))
    assert first.rows[0]['video_path'] == str(media / 'v/x.mp4')
    media = move_root('second')
    second = VALeJEPADataset(dict(datasets=[str(split)]))
    assert first.rows.directory != second.rows.directory
    assert second.rows[0]['audio_path'] == str(media / 'a/x.wav')
    with pytest.raises(ValueError, match='Unknown dataset'):
        paths.get_dataset_paths([str(tmp_path / 'missing.csv')])


def test_all_task_datasets_are_registered():
    import yaml
    registry = paths.dataset_roots()
    for config in (paths.PROJECT_ROOT / 'CONFIGS/data/finetune').rglob('*.yaml'):
        data = yaml.safe_load(config.read_text())['data']
        assert 'rootpaths' not in data
        assert 'manifests' not in data
        for dataset in data.get('datasets', []):
            assert str((paths.PROJECT_ROOT / dataset).resolve()) in registry
