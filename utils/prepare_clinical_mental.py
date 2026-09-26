"""Refresh the clinical split and its 38 mental finetuning configurations.

Run from the repository root: python -m utils.prepare_clinical_mental
"""
import argparse
import csv
import hashlib
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/home/data/sdb/基座模型数据集/3心理数据集/临床/split-0901.csv')
SPLIT = Path('DATASET/splits-0901/3_私有临床数据_LV.csv')
PAIRED = Path('DATASET/merged/mental/clinical_canonical.csv')


def prepare(source):
    source = source.resolve()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    tasks = [field[:-6] for field in fields if field.endswith('_split')]
    if len(tasks) != 38 or not rows:
        raise ValueError('Expected a nonempty clinical CSV with 38 task split columns')
    classes = {}
    for task in tasks:
        valid = [row for row in rows if row[task + '_split'] not in {'', '-1'}]
        labels = {int(row[task]) for row in valid}
        if labels != set(range(1, max(labels) + 1)):
            raise ValueError(f'{task}: expected contiguous one-based class labels')
        splits = Counter(row[task + '_split'] for row in valid)
        if set(splits) != {'0', '1'}:
            raise ValueError(f'{task}: expected training split 0 and validation split 1')
        classes[task] = max(labels)

    paired_rows, seen = [], set()
    for row in rows:
        video, audio = [str((source.parent / row[key]).resolve()) for key in ('video_path', 'audio_path')]
        source_id = Path(video).stem
        if not row['video_path'] or not row['audio_path'] or source_id != Path(audio).stem:
            raise ValueError('Clinical audio/video must have matching nonempty media identities')
        pair_id = hashlib.sha256(f'{video}\0{audio}'.encode()).hexdigest()[:24]
        if pair_id in seen:
            raise ValueError(f'Duplicate media pair: {video}')
        seen.add(pair_id)
        paired_rows.append(dict(row, video_path=video, audio_path=audio, pair_id=pair_id, source_id=source_id))

    split_path = PROJECT_ROOT / SPLIT
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if source != split_path.resolve():
        shutil.copyfile(source, split_path)
    paired_path = PROJECT_ROOT / PAIRED
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    with paired_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['pair_id', 'source_id', *fields])
        writer.writeheader()
        writer.writerows(paired_rows)

    catalog = PROJECT_ROOT / 'CONFIGS/datasets.csv'
    with catalog.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        catalog_fields, entries = reader.fieldnames, list(reader)
    for entry in entries:
        if entry['split_csv'] == SPLIT.name:
            entry.update(root_path=str(source.parent) + '/', original_csv_path=str(source))
    with catalog.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=catalog_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(entries)

    for modality, app in [('audio', 'finetune_a'), ('video', 'finetune_v'), ('audio-video', 'finetune_va')]:
        template_name = 'RAVDESS-emotion'
        template_path = PROJECT_ROOT / f'CONFIGS/tasks/finetune/{modality}/classification/{template_name}.yaml'
        template = yaml.safe_load(template_path.read_text())
        for task in tasks:
            data_path = Path(f'CONFIGS/data/finetune/{modality}/mental/{task}.yaml')
            data = dict(task='classification', num_class=classes[task], label_column=task)
            if modality == 'audio-video':
                data['manifests'] = [str(PAIRED)]
            else:
                data.update(datasets=[str(SPLIT)], rootpaths=[str(source.parent) + '/'])
                if modality == 'video':
                    data['datasets_weights'] = [1]
            config = deepcopy(template)
            config['folder'] = f'OUTPUT/{app}/mental/{task}'
            if modality == 'audio-video':
                config['data'] = {'local_views': 0}
                config['finetune'].update(best_metric='f1_macro', best_metric_mode='max', branch_loss_weight=0.0)
                # Keep task-specific data last and use the shared sampling preset.
                config['yamls'].pop('data', None)
                config['yamls']['sampling'] = 'CONFIGS/data/finetune/audio-video/sampling/va-6s.yaml'
            config['yamls']['data'] = str(data_path)
            task_path = Path(f'CONFIGS/tasks/finetune/{modality}/mental/{task}.yaml')
            for path, contents in [(data_path, {'data': data}), (task_path, config)]:
                target = PROJECT_ROOT / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(yaml.safe_dump(contents, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'Replaced {SPLIT}: {len(rows)} rows; wrote {PAIRED}; generated 38 tasks × 3 modalities.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    prepare(parser.parse_args().source)
