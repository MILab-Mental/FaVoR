"""Shared paths for baseline training and reporting."""
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


with (ROOT / 'BASELINES/paths.yaml').open(encoding='utf-8') as stream:
    MACHINE_PATHS = {
        name: path(os.environ.get(name) or value)
        for name, value in yaml.safe_load(stream).items()
    }

OUTPUT_ROOT = MACHINE_PATHS['BASELINES_OUTPUT_ROOT']
WEIGHTS_ROOT = MACHINE_PATHS['BASELINES_WEIGHTS_ROOT']
