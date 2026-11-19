import csv
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
