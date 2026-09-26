"""Configuration moves must preserve old launch commands and saved includes."""
import csv
from pathlib import Path

import pytest
import yaml

from app import main


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'CONFIGS/path-mapping.csv').open(newline='', encoding='utf-8') as mapping_file:
    CONFIG_PATHS = [row for row in csv.DictReader(mapping_file) if row['old_path'].endswith('.yaml')]


@pytest.mark.parametrize('paths', CONFIG_PATHS, ids=lambda row: row['old_path'])
def test_legacy_and_current_configs_load_identically(paths):
    assert main.load_config(ROOT / paths['old_path']) == main.load_config(ROOT / paths['new_path'])


def test_saved_config_can_include_old_fragment_path(tmp_path):
    fragment = next(row for row in CONFIG_PATHS if row['old_path'] == 'CONFIGS/opt/afinetune-opt-30.yaml')
    snapshot = tmp_path / 'saved.yaml'
    snapshot.write_text(yaml.safe_dump({'yamls': [fragment['old_path']], 'folder': 'OUTPUT/existing-run'}))
    expected = main.load_config(ROOT / fragment['new_path'])
    expected['folder'] = 'OUTPUT/existing-run'
    assert main.load_config(snapshot) == expected


def test_existing_file_takes_priority_over_migration_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    config_dir = tmp_path / 'CONFIGS'
    config_dir.mkdir()
    (config_dir / 'path-mapping.csv').write_text('old_path,new_path\nCONFIGS/old.yaml,CONFIGS/new.yaml\n')
    (config_dir / 'old.yaml').write_text('value: original\n')
    (config_dir / 'new.yaml').write_text('value: migrated\n')
    assert main.load_config(config_dir / 'old.yaml') == {'value': 'original'}
