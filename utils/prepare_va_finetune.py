"""Prepare paired manifests for catalogued audio-video downstream tasks.

Run from the repository root: python -m utils.prepare_va_finetune
Refresh the clinical data first when needed with utils.prepare_clinical_mental.
"""
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

from datasets.common.paths import get_dataset_paths
from datasets.va_lejepa.va_manifest import source_key, validate_identity
from utils.finetune_catalog import read_tasks, task_data


ROOT = Path(__file__).resolve().parents[1]


def prepare():
    manifests = {}
    summary = []
    for entry in read_tasks(ROOT):
        if 'va' not in entry['modalities'].split('|'):
            continue
        original = task_data(entry)
        media_roots = get_dataset_paths(original['datasets'])
        all_rows = []
        for csv_name, media_root in zip(original['datasets'], media_roots):
            key = (csv_name, media_root)
            if key not in manifests:
                with (ROOT / csv_name).open(encoding='utf-8-sig', newline='') as handle:
                    reader = csv.DictReader(handle)
                    fields = reader.fieldnames
                    rows, seen = [], set()
                    for line, source in enumerate(reader, 2):
                        if not source.get('audio_path') or not source.get('video_path'):
                            continue
                        video, audio = [str((Path(media_root) / source[field]).absolute())
                                        for field in ('video_path', 'audio_path')]
                        row = dict(source, video_path=video, audio_path=audio,
                                   source_id=source_key(video),
                                   pair_id=hashlib.sha256(f'{video}\0{audio}'.encode()).hexdigest()[:24])
                        try:
                            validate_identity(row)
                        except ValueError as error:
                            raise ValueError(f'{csv_name}:{line}: {error}') from error
                        if row['pair_id'] in seen:
                            raise ValueError(f'{csv_name}:{line}: duplicate pair')
                        seen.add(row['pair_id'])
                        rows.append(row)
                if not rows:
                    raise ValueError(f'{csv_name}: no paired audio/video rows')
                relative = (Path('DATASET/merged/mental/clinical_canonical.csv')
                            if entry['scope'] == 'mental'
                            else Path('DATASET/merged/finetune_va') / Path(csv_name).name)
                target = ROOT / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['pair_id', 'source_id', *fields])
                    writer.writeheader()
                    writer.writerows(rows)
                manifests[key] = rows
            all_rows.extend(manifests[key])

        task, label = original['task'], original['label_column']
        classes = original['num_class']
        counts = Counter()
        for row in all_rows:
            split = str(row[label + '_split']).strip()
            if split in {'', '-1'}:
                continue
            if split not in {'0', '1'}:
                raise ValueError(f'{entry["task_id"]}: invalid split {split}')
            value = str(row[label]).strip()
            if task == 'regression':
                valid = math.isfinite(float(value))
            else:
                tokens = value.split('|') if task == 'multi_label_classification' else [value]
                valid = all(1 <= int(token) <= classes for token in tokens)
            if not valid:
                raise ValueError(f'{entry["task_id"]}: invalid target {value}')
            counts[split] += 1
        if not counts['0'] or not counts['1']:
            raise ValueError(f'{entry["task_id"]}: missing paired training or validation samples')
        group = ('mental' if entry['scope'] == 'mental' else
                 {'classification': 'classification', 'regression': 'regression',
                  'multi_label_classification': 'multilabel'}[task])
        summary.append(dict(group=group, task=entry['task_id'],
                            train=counts['0'], validation=counts['1']))
    report = ROOT / 'DATASET/merged/finetune_va/tasks.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['group', 'task', 'train', 'validation'])
        writer.writeheader()
        writer.writerows(summary)
    print(f'Prepared {len(summary)} VA tasks across {len(manifests)} paired datasets; split counts: {report}')


if __name__ == '__main__':
    prepare()
