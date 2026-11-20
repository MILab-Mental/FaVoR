import csv
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nonmental_tasks_use_single_sentence_profile():
    with (ROOT / 'CONFIGS/tasks/finetune/tasks.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['profile'] == 'clip' for row in rows if row['scope'] == 'nonmental')
    assert all(row['profile'] == 'instance' for row in rows if row['scope'] == 'mental')


def test_launcher_retries_incomplete_and_continues_after_failure(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    dataset = tmp_path / 'data.csv'
    checkpoint.touch()
    dataset.touch()
    catalog = tmp_path / 'tasks.csv'
    fields = ('enabled,task_id,dataset,scope,task_type,num_class,csv_path,label_column,'
              'modalities,profile,output_dir,best_metric,best_metric_mode,overrides').split(',')
    with catalog.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ('first', 'second', 'third'):
            writer.writerow(dict(enabled=1, task_id=task, dataset='test', scope='nonmental',
                                 task_type='classification', num_class=2, csv_path=dataset,
                                 label_column=task, modalities='a', profile='clip',
                                 output_dir=str(tmp_path / 'outputs' / '{task_id}' / '{run}'),
                                 best_metric='f1_macro', best_metric_mode='max', overrides=''))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    torchrun = bin_dir / 'torchrun'
    torchrun.write_text('''#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in folder=*) folder=${arg#folder=};; data.label_column=*) task=${arg#data.label_column=};; esac
done
mkdir -p "$folder/logs"
printf '%s\\n' "$task" >> "$CALLS"
printf result > "$folder/best.pt"
printf result > "$folder/logs/history.csv"
printf result > "$folder/logs/best.csv"
if [[ $task == second && ! -f $RETRY ]]; then exit 7; fi
''')
    torchrun.chmod(0o755)
    calls = tmp_path / 'calls'
    retry = tmp_path / 'retry'
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'TASK_CSV': str(catalog),
           'CHECKPOINT': str(checkpoint), 'RUN_NAME': 'run', 'CALLS': str(calls),
           'RETRY': str(retry), 'CUDA_VISIBLE_DEVICES': '0'}

    def run(**overrides):
        return subprocess.run(['bash', str(ROOT / 'train_finetune.sh'), 'a'], cwd=ROOT,
                              env={**env, **overrides}, text=True, capture_output=True)

    first = run()
    assert first.returncode != 0
    assert calls.read_text().splitlines() == ['first', 'second', 'third']
    outputs = tmp_path / 'outputs'
    assert (outputs / 'first/run/.finetune-complete').is_file()
    assert not (outputs / 'second/run/.finetune-complete').exists()
    assert (outputs / 'third/run/.finetune-complete').is_file()

    preview = run(DRY_RUN='1')
    assert preview.returncode == 0
    assert 'Would preserve incomplete output' in preview.stdout
    assert not list((outputs / 'second').glob('run.incomplete-*'))
    retry.touch()
    assert run().returncode == 0
    assert calls.read_text().splitlines() == ['first', 'second', 'third', 'second']
    assert (outputs / 'second/run/.finetune-complete').is_file()
    assert len(list((outputs / 'second').glob('run.incomplete-*'))) == 1


def test_mental_auto_route_and_checkpoint_override(tmp_path):
    checkpoint = tmp_path / 'custom.pt'
    checkpoint.touch()
    dataset = tmp_path / 'dataset.csv'
    dataset.touch()
    source = ROOT / 'CONFIGS/tasks/finetune/tasks.csv'
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        task = next(row for row in reader if row['task_id'] == 'DSM-Depression')
    task['csv_path'] = str(dataset)
    catalog = tmp_path / 'tasks.csv'
    with catalog.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(task)
    env = {**os.environ, 'MENTAL_AUTO': 'depression-best', 'SCOPE': 'mental',
           'TASKS': 'DSM-Depression', 'DRY_RUN': '1', 'CHECKPOINT': str(checkpoint),
           'RUN_NAME': 'custom', 'CUDA_VISIBLE_DEVICES': '0', 'TASK_CSV': str(catalog)}
    command = ['bash', str(ROOT / 'train_finetune.sh')]
    result = subprocess.run(command + ['va'], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'mental-auto/depression-best.yaml' in result.stdout
    assert f'finetune.pretrained_checkpoint={checkpoint}' in result.stdout
    assert 'mental-transfer/DSM-Depression/custom' in result.stdout
    wrong_mode = subprocess.run(command + ['a'], cwd=ROOT, env=env, text=True, capture_output=True)
    assert wrong_mode.returncode != 0
    assert 'requires modality va' in wrong_mode.stderr
    wrong_scope = subprocess.run(command + ['va'], cwd=ROOT, env={**env, 'SCOPE': 'all'},
                                 text=True, capture_output=True)
    assert wrong_scope.returncode != 0
    assert 'requires SCOPE=mental' in wrong_scope.stderr


def test_va_route_selects_ordinary_tasks_and_preserves_profiles(tmp_path):
    checkpoint = tmp_path / 'custom.pt'
    dataset = tmp_path / 'dataset.csv'
    checkpoint.touch()
    dataset.touch()
    with (ROOT / 'CONFIGS/tasks/finetune/tasks.csv').open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    catalog = tmp_path / 'tasks.csv'
    with catalog.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row, csv_path=str(dataset)) for row in rows)
    env = {**os.environ, 'VA_ROUTE': 'total-best', 'MENTAL_AUTO': '', 'SCOPE': 'nonmental',
           'TASKS': '', 'TYPE': 'all', 'DRY_RUN': '1', 'CHECKPOINT': str(checkpoint),
           'RUN_NAME': 'total-best-b1', 'CUDA_VISIBLE_DEVICES': '0', 'TASK_CSV': str(catalog)}
    result = subprocess.run(['bash', str(ROOT / 'train_finetune.sh'), 'va',
                             'data.batch_size=1', f'folder={tmp_path / "output"}'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('torchrun ')]
    assert len(commands) == 15
    for command in commands:
        assert 'mental-auto/total-best.yaml' in command[command.index('--fname') + 1]
        args = command[command.index('--set') + 1:]
        assert 'data.batch_size=1' in args
        assert 'data.num_sentence=0' in args
        if 'data.task=regression' in args:
            assert 'finetune.best_metric=mae' in args
