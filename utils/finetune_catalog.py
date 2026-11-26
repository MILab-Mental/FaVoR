"""Read the shared task definitions used by fine-tuning and data preparation."""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_tasks(root=ROOT):
    with (root / 'CONFIGS/tasks/finetune/tasks.csv').open(encoding='utf-8-sig', newline='') as handle:
        return [row for row in csv.DictReader(handle) if row['enabled'] == '1']


def mental_core_tasks(root=ROOT):
    return [row['task_id'] for row in read_tasks(root)
            if row['scope'] == 'mental' and row.get('mental_core') == '1']


def task_data(row):
    return {'datasets': [row['csv_path']], 'datasets_weights': [1],
            'task': row['task_type'], 'num_class': int(row['num_class']),
            'label_column': row['label_column']}
