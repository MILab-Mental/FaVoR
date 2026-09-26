"""Prepare paired manifests and concrete configurations for all VA downstream tasks.

Run from the repository root: python -m utils.prepare_va_finetune
Requires the existing DATASET/splits-0901 CSVs (refresh clinical data first with
python -m utils.prepare_clinical_mental when needed).
"""
import csv
import hashlib
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from datasets.va_lejepa.va_manifest import source_key, validate_identity


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('classification', 'multilabel', 'regression', 'mental')


def write_yaml(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')


def prepare():
    template = yaml.safe_load((ROOT / 'CONFIGS/tasks/finetune/audio-video/classification/RAVDESS-emotion.yaml').read_text())
    manifests = {}
    summary = []
    for group in GROUPS:
        for fragment in sorted((ROOT / f'CONFIGS/data/finetune/video/{group}').glob('*.yaml')):
            original = yaml.safe_load(fragment.read_text())['data']
            if len(original['datasets']) != len(original['rootpaths']):
                raise ValueError(f'{fragment}: datasets/rootpaths length mismatch')
            paired_paths, all_rows = [], []
            for csv_name, media_root in zip(original['datasets'], original['rootpaths']):
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
                    relative = (Path('DATASET/merged/mental/clinical_canonical.csv') if group == 'mental'
                                else Path('DATASET/merged/finetune_va') / Path(csv_name).name)
                    target = ROOT / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('w', encoding='utf-8', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['pair_id', 'source_id', *fields])
                        writer.writeheader()
                        writer.writerows(rows)
                    manifests[key] = (str(relative), rows)
                relative, rows = manifests[key]
                paired_paths.append(relative)
                all_rows.extend(rows)

            task, label = original['task'], original['label_column']
            classes = int(original.get('num_class', 1))
            counts = Counter()
            for row in all_rows:
                split = str(row[label + '_split']).strip()
                if split in {'', '-1'}:
                    continue
                if split not in {'0', '1'}:
                    raise ValueError(f'{fragment}: invalid split {split}')
                value = str(row[label]).strip()
                if task == 'regression':
                    valid = math.isfinite(float(value))
                else:
                    tokens = value.split('|') if task == 'multi_label_classification' else [value]
                    valid = all(1 <= int(token) <= classes for token in tokens)
                if not valid:
                    raise ValueError(f'{fragment}: invalid target {value}')
                counts[split] += 1
            if not counts['0'] or not counts['1']:
                raise ValueError(f'{fragment}: missing paired training or validation samples')
            data_path = Path(f'CONFIGS/data/finetune/audio-video/{group}/{fragment.name}')
            data = dict(task=task, label_column=label, num_class=classes, manifests=paired_paths)
            write_yaml(ROOT / data_path, {'data': data})
            config = deepcopy(template)
            config.update(app='finetune_va', folder=f'OUTPUT/finetune_va/{group}/{fragment.stem}',
                          data={'local_views': 0})
            config['finetune'].update(best_metric='mae' if task == 'regression' else 'f1_macro',
                                      best_metric_mode='min' if task == 'regression' else 'max',
                                      branch_loss_weight=0.0)
            config['yamls'] = dict(
                sampling='CONFIGS/data/finetune/audio-video/sampling/va-6s.yaml',
                model='CONFIGS/models/finetune/audio-video/vitl-emotion2vec.yaml',
                opt='CONFIGS/optimization/finetune/audio-video/epochs20-warmup0-lr0.001.yaml',
                data=str(data_path))
            write_yaml(ROOT / f'CONFIGS/tasks/finetune/audio-video/{group}/{fragment.name}', config)
            summary.append(dict(group=group, task=fragment.stem, train=counts['0'], validation=counts['1']))
    report = ROOT / 'DATASET/merged/finetune_va/tasks.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['group', 'task', 'train', 'validation'])
        writer.writeheader()
        writer.writerows(summary)
    print(f'Prepared {len(summary)} VA tasks across {len(manifests)} paired datasets; split counts: {report}')


if __name__ == '__main__':
    prepare()
