"""One entry for training, evaluation, resume and bounded real-data checks."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import load_config
from BASELINES.common.runtime import ROOT, OUTPUT_ROOT, path, load_file, write_config, write_json,apply_overrides


def verify(stage, config, experiment, device, extra=()):
    cfg = load_config(config)
    run_name = experiment or cfg['experiment']+'-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output = OUTPUT_ROOT / stage / 'smoke' / cfg['model']['id'] / run_name
    checkpoint_name='latest.pt' if stage=='pretrain' else 'best.pt'
    if output.exists():
        raise FileExistsError(f'{output}: choose a new --experiment name')
    base = [sys.executable, '-m', 'BASELINES.run', '--stage', stage, '--config', str(config)]
    overrides = [*extra, f'experiment={run_name}', f'output={output}', f'device={device}', 'data.limit=2',
                 'data.batch_size=2', 'max_batches=1', 'optimization.epochs=2']
    def run(arguments, suffix):
        output.mkdir(parents=True, exist_ok=True)
        command = base + arguments
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            (output / 'smoke-failure.log').write_text(result.stdout)
            raise RuntimeError(f'{cfg["model"]["id"]}: {suffix} failed; see {output}/smoke-failure.log')
    run(['--set', *overrides, 'stop_after_epoch=1'], 'first update')
    first = load_file(output / checkpoint_name)
    if first['step'] != 1 or not first['config'].get('gradient_check', {}).get('encoder'):
        raise AssertionError('First real update did not reach the encoder')
    del first
    run(['--resume', str(output / checkpoint_name), '--set', *overrides, 'stop_after_epoch=2'], 'resume')
    last = load_file(output / checkpoint_name)
    report=json.loads((output/'metrics.json').read_text())
    # A validation-selected best.pt may correctly stay at epoch 1 even after
    # resumed training completes epoch 2. The run report tracks actual progress.
    if report['optimizer_steps'] != 2 or report['history'][-1]['epoch'] != 2:
        raise AssertionError('Resume did not advance optimizer/epoch state')
    result = dict(model=cfg['model']['id'], configuration=str(config),
                  stage=stage, route=cfg['route'], output=str(output),
                  complete_resume_verified=True, optimizer_steps=report['optimizer_steps'],
                  csv_splits=list(last['config']['data_audit']['splits']),
                  parameters=last['config']['parameter_counts'], gradient_check=last['config']['gradient_check'],
                  real_manifests=last['config']['data_audit']['manifests'])
    del last
    for split in (s for s in ('validation', 'test') if s in result['csv_splits']):
        run(['--evaluate', split, '--checkpoint', str(output / checkpoint_name), '--set',
             *overrides, 'save_predictions=true'], split)
    if stage == 'finetune' and 'test' in result['csv_splits']:
        report = json.loads((output / 'metrics.json').read_text())
        result['test_samples'] = report['evaluations']['test']['metrics']['samples']
        import csv
        with (output / 'predictions.csv').open(newline='') as handle:
            if sum(1 for _ in csv.DictReader(handle)) != result['test_samples']:
                raise AssertionError('Prediction CSV row count mismatch')
    return result


def remove_smoke_checkpoints(result):
    """Keep the evidence after restore/transfer checks, discard disposable weights."""
    output = path(result['output'])
    for name in (('latest.pt',) if result['stage']=='pretrain' else ('best.pt',)):
        (output / name).unlink()
    cfg = load_config(output / 'config.yaml')
    cfg['artifacts'] = dict(checkpoints_retained=False,
        note='Temporary smoke checkpoints removed after resume/evaluation/transfer checks.')
    write_config(output, cfg)
    metrics = json.loads((output / 'metrics.json').read_text())
    metrics['artifacts'] = dict(checkpoints_retained=False)
    write_json(output / 'metrics.json', metrics)
    result['checkpoints_retained'] = False


def main():
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument('--stage', choices=['finetune', 'pretrain'], default='finetune')
    parser = argparse.ArgumentParser(parents=[stage_parser], description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--models', nargs='+', help='Configuration names without .yaml; all selects every config in this stage')
    source.add_argument('--config', help='Custom experiment YAML, including task/data settings')
    parser.add_argument('--list', action='store_true', help='List stage configurations without building models')
    parser.add_argument('--gpus',type=int,default=1,help='Use 2 for the full DDP comparison matrix')
    parser.add_argument('--tasks',nargs='+',help='all, ordinary, mental, or explicit task names')
    parser.add_argument('--test',action='store_true',help='Real training checks entirely in RAM; no files, downloads or cache writes')
    parser.add_argument('--device', help='Override device, e.g. cuda:1 or cpu')
    parser.add_argument('--experiment', help='Run directory name; smoke defaults to configured task/route plus timestamp')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=YAML')
    parser.add_argument('--resume', help='Restore latest.pt (pretraining) or an unfinished best.pt (finetuning)')
    parser.add_argument('--evaluate', choices=['validation', 'test'])
    parser.add_argument('--checkpoint', help='Checkpoint for evaluation')
    parser.add_argument('--smoke', action='store_true', help='Two updates, full resume, evaluation, and pretrain-to-finetune transfer')
    parser.add_argument('--finetune-config', help='Custom downstream config for a pretraining smoke check')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Keep temporary latest.pt (pretraining) or best.pt (finetuning) after --smoke')
    args = parser.parse_args()
    directory = ROOT / 'BASELINES' / args.stage / 'configs'
    available = sorted(p for p in directory.glob('*.yaml') if p.stem != 'defaults')
    if args.list:
        for config in available:
            cfg = load_config(config)
            print(f'{config.stem:55} {cfg["route"]:22} {cfg["model"]["kind"]}')
        return
    if args.config:
        configs = [path(args.config)]
    elif args.models == ['all']:
        configs = available
    elif args.models:
        configs = [directory / f'{name}.yaml' for name in args.models]
    elif args.smoke:
        names = ['wavlm-large', 'videomae-vit-large-patch16', 'cav-mae-scale-plus-plus'] if args.stage=='finetune' else ['audiomae-large', 'videomae-vit-large-patch16', 'leavjepa-large']
        configs = [directory / f'{name}.yaml' for name in names]
    else:
        parser.error('Choose --models or --config; use --list to inspect configurations')
    if args.test or args.tasks or args.gpus>1:
        from BASELINES.common.launch import matrix
        return matrix(configs,args)
    if (args.resume or args.evaluate or args.checkpoint) and (args.smoke or len(configs)!=1):
        parser.error('Resume/evaluation requires one non-smoke experiment')
    if bool(args.evaluate) != bool(args.checkpoint):
        parser.error('--evaluate and --checkpoint must be used together')
    if args.resume and args.evaluate:
        parser.error('Choose either --resume or --evaluate')
    if args.keep_checkpoints and not args.smoke:
        parser.error('--keep-checkpoints only applies to --smoke; training keeps latest.pt (pretraining) or best.pt (finetuning)')
    if args.finetune_config and not (args.smoke and args.stage=='pretrain'):
        parser.error('--finetune-config requires --stage pretrain --smoke')
    if not args.smoke:
        from BASELINES.common.launch import python_environment
        for config in configs:
            cfg=apply_overrides(load_config(config),args.set)
            command = [cfg.get('python',sys.executable), '-B', '-m', f'BASELINES.{args.stage}.main', '--config', str(config)]
            for key in ('resume', 'evaluate', 'checkpoint'):
                if getattr(args, key):
                    command.extend([f'--{key}', getattr(args, key)])
            overrides = list(args.set)
            for key in ('device', 'experiment'):
                if getattr(args, key):
                    overrides.append(f'{key}={getattr(args, key)}')
            if overrides:
                command.extend(['--set', *overrides])
            result = subprocess.run(command, cwd=ROOT,env=python_environment(command[0]))
            if result.returncode:
                sys.exit(result.returncode)
        return
    experiment = args.experiment
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for config in configs:
        try:
            checked = [verify(args.stage, config, experiment, device, args.set)]
            if args.stage == 'pretrain':
                checkpoint = str(path(checked[0]['output']) / 'latest.pt')
                finetune = checked[0]['model']+('-continued' if checked[0]['route']=='continue_pretrain' else '')
                downstream = path(args.finetune_config) if args.finetune_config else ROOT / 'BASELINES/finetune/configs' / f'{finetune}.yaml'
                checked.append(verify('finetune', downstream, experiment, device,
                                      [f'model.encoder_checkpoint={checkpoint}']))
            for result in checked:
                result['checkpoints_retained'] = True
                if not args.keep_checkpoints:
                    remove_smoke_checkpoints(result)
                report = path(result['output']) / 'metrics.json'
                metrics = json.loads(report.read_text())
                metrics['smoke_check'] = dict(
                    complete_resume_verified=True, optimizer_steps=result['optimizer_steps'],
                    encoder_transfer_verified=args.stage=='pretrain',
                    checkpoints_retained=result['checkpoints_retained'])
                write_json(report, metrics)
            results.extend(checked)
            print(config.stem, 'passed', flush=True)
        except Exception as exc:
            results.append(dict(configuration=str(config), status='failed', reason=str(exc)))
            print(config.stem, 'failed:', exc, flush=True)
    if any(result.get('status') == 'failed' for result in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
