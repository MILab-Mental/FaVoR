"""Align existing FAVOR A/V predictions and select probability weight on validation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from app.main import load_config
from datasets.common.instances import sentence_identity
from datasets.va_lejepa.va_manifest import source_key
from BASELINES.common.data import read_rows, partition, configure_instances
from BASELINES.common.runtime import setup, write_config, write_json, path, fingerprint
from BASELINES.finetune.trainer import metrics_for
from utils.classification_metrics import write_predictions_csv, write_multilabel_predictions_csv, write_regression_predictions_csv


def combine(audio, video, weight, task):
    values = weight*audio+(1-weight)*video
    if task == 'classification':
        return np.log(np.clip(values, 1e-12, 1))
    if task == 'multi_label_classification':
        p = np.clip(values, 1e-7, 1-1e-7)
        return np.log(p/(1-p))
    return values[:, None]


def load_predictions(filename, lookup, allowed, task, classes):
    values = {}
    with path(filename).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            filename = row['path']
            if filename not in lookup:
                raise ValueError(f'Prediction path is outside the audited paired manifest: {filename}')
            key = lookup[filename]
            if key not in allowed or key in values:
                raise ValueError('Wrong split or duplicate instance in prediction CSV')
            if task == 'regression':
                truth, prediction = float(row['target']), float(row['prediction'])
            else:
                truth = int(row['true_label']) if task == 'classification' else np.array([int(str(i+1) in row['true_label'].split('|')) for i in range(classes)])
                prediction = np.array([float(row[f'prob_{i}']) for i in range(classes)])
                if not np.isfinite(prediction).all() or np.any((prediction<0) | (prediction>1)):
                    raise ValueError('Invalid source probabilities')
                if task == 'classification' and not np.isclose(prediction.sum(), 1, atol=1e-5):
                    raise ValueError('Source class probabilities do not sum to one')
            labels_match = np.isclose(truth, allowed[key], rtol=0, atol=1e-6) if task=='regression' else np.array_equal(truth, allowed[key])
            if not labels_match or not np.isfinite(prediction).all():
                raise ValueError('Source prediction labels differ from task manifest')
            values[key] = prediction
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-config', required=True, help='A baseline config carrying the audited task/group protocol')
    for mode in ('audio', 'video'):
        for split in ('validation', 'test'):
            parser.add_argument(f'--{mode}-{split}', required=split=='test')
        parser.add_argument(f'--{mode}-run-config', help='Saved run config with measured parameter_counts')
    parser.add_argument('--experiment', default='favor-late-fusion')
    parser.add_argument('--audio-weight', type=float, default=.5, help='Fixed probability weight when the CSV has no validation split')
    args = parser.parse_args()
    cfg = load_config(path(args.data_config))
    cfg.update(stage='finetune', experiment=args.experiment, model=dict(id='favor-av-late-fusion'))
    cfg.pop('output', None)
    output, logger, _ = setup(cfg)
    configure_instances(cfg, 'audio-video')
    data = cfg['data']
    data['cohort'] = 'paired'
    rows, _ = read_rows(data, 'audio-video')
    splits = partition(rows, data)
    if 'test' not in splits:
        raise ValueError('The CSV has no test split')
    if not 0 <= args.audio_weight <= 1:
        raise ValueError('--audio-weight must be between zero and one')
    if 'validation' not in splits and (args.audio_validation or args.video_validation):
        raise ValueError('The CSV has no validation split; use a fixed --audio-weight')
    task = data['task']
    classes = 1 if task == 'regression' else int(data['num_class'])
    lookup, allowed = {}, {}
    for split, records in splits.items():
        allowed[split] = {}
        for row in records:
            filename = row['video_path']
            instance = sentence_identity(filename)[0] if data.get('num_sentence') else filename
            key = row['_manifest'] + ':' + source_key(instance)
            if key in allowed[split] and not np.array_equal(allowed[split][key], row['_label']):
                raise ValueError('Inconsistent instance labels')
            allowed[split][key] = row['_label']
            for field in ('audio_path', 'video_path'):
                value = row[field]
                if data.get('num_sentence'):
                    value = sentence_identity(value)[0]
                lookup[value] = key
    paired = {}
    sources = {}
    for split in (s for s in ('validation', 'test') if s in splits):
        a_path, v_path = getattr(args, 'audio_'+split), getattr(args, 'video_'+split)
        if not a_path or not v_path:
            raise ValueError(f'Both modalities need {split} predictions for the CSV split')
        audio = load_predictions(a_path, lookup, allowed[split], task, classes)
        video = load_predictions(v_path, lookup, allowed[split], task, classes)
        if set(audio) != set(video) or set(audio) != set(allowed[split]):
            raise ValueError('Both modalities must predict exactly the audited common cohort; missing predictions cannot be dropped silently')
        keys = sorted(audio)
        paired[split] = (keys, np.asarray([allowed[split][k] for k in keys]),
                         np.asarray([audio[k] for k in keys]), np.asarray([video[k] for k in keys]))
        sources[split] = dict(audio=dict(path=a_path, sha256=fingerprint(a_path)), video=dict(path=v_path, sha256=fingerprint(v_path)))
    best = dict(audio_weight=args.audio_weight, selection='fixed_weight_no_csv_validation')
    metric = 'mae' if task == 'regression' else 'f1_macro'
    if 'validation' in paired:
        _, truth, audio, video = paired['validation']
        best = None
        for weight in np.linspace(0, 1, 21):
            metrics, _, _ = metrics_for(task, truth, combine(audio, video, weight, task), classes)
            score = -metrics[metric] if task == 'regression' else metrics[metric]
            if best is None or score > best['score']:
                best = dict(audio_weight=float(weight), score=float(score), metric=metric, validation=metrics, selection='csv_validation')
    keys, truth, audio, video = paired['test']
    logits = combine(audio, video, best['audio_weight'], task)
    metrics, predictions, probabilities = metrics_for(task, truth, logits, classes)
    if task == 'classification':
        write_predictions_csv(output/'predictions.csv', keys, truth, predictions, logits, probabilities)
    elif task == 'multi_label_classification':
        write_multilabel_predictions_csv(output/'predictions.csv', keys, truth, predictions, logits, probabilities)
    else:
        write_regression_predictions_csv(output/'predictions.csv', keys, truth, predictions)
    counts = {}
    for mode in ('audio', 'video'):
        config_path = getattr(args, mode+'_run_config')
        if config_path:
            counts[mode] = yaml.safe_load(path(config_path).read_text()).get('parameter_counts')
    cfg.update(sources=sources, source_parameter_counts=counts,
               count_scope='Sum the two measured downstream models; fusion adds zero parameters. Missing source counts remain unavailable.')
    write_config(output, cfg)
    write_json(output/'metrics.json', dict(best=best, test=metrics, samples=len(keys), source_parameter_counts=counts))
    torch.save(dict(best=best, config=cfg), output/'best.pt')
    logger.info('selection=%s audio_weight=%.2f test_instances=%d', best['selection'], best['audio_weight'], len(keys))


if __name__ == '__main__':
    main()
