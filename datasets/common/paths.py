"""Resolve dataset media roots from the shared registry."""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_CSV = PROJECT_ROOT / 'CONFIGS/datasets.csv'


def _absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def dataset_roots():
    with DATASETS_CSV.open(encoding='utf-8-sig', newline='') as handle:
        roots = {}
        for row in csv.DictReader(handle):
            key = str(_absolute(row['split_csv']))
            if key in roots:
                raise ValueError(f'Duplicate dataset in {DATASETS_CSV}: {row["split_csv"]}')
            roots[key] = str(_absolute(row['root_path']))
        return roots


def get_dataset_paths(datasets):
    roots = dataset_roots()
    paths = []
    for dataset in datasets:
        key = str(_absolute(dataset))
        if key not in roots:
            raise ValueError(f'Unknown dataset {dataset}: add it to {DATASETS_CSV}')
        paths.append(roots[key])
    return paths
