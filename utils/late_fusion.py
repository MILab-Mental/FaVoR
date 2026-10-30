"""Align existing FAVOR A/V predictions and select probability weight on validation."""
import argparse
import csv
import hashlib
import logging
import json
from pathlib import Path

import numpy as np
import yaml

from app.main import load_config
from datasets.common.instances import sentence_identity
from datasets.va_lejepa.va_manifest import source_key, iter_manifest
from datasets.common.paths import get_dataset_paths
from datasets.audio_jepa.finetune_dataset import AudioCSVDataset
from utils.instance_finetune_configs import is_long_media_task
from app.finetune_a.train import _regression_metrics
from utils.classification_metrics import classification_metrics, multilabel_metrics
from utils.classification_metrics import write_predictions_csv, write_multilabel_predictions_csv, write_regression_predictions_csv


ROOT=Path(__file__).resolve().parents[1]


def path(value):
    value=Path(value).expanduser()
    return value if value.is_absolute() else ROOT/value


def fingerprint(filename):
    digest=hashlib.sha256()
    with path(filename).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def paired_splits(data):
    """Native paired manifest/labels and original CSV assignments; no decoding."""
    parser=object.__new__(AudioCSVDataset)
    parser.task,parser.num_class=data['task'],int(data.get('num_class',1))
    manifests=data['datasets']
    roots=data.get('rootpaths') or get_dataset_paths(manifests)
    if len(roots)!=len(manifests):raise ValueError('datasets/rootpaths length mismatch')
    splits={}
    for manifest,root in zip(manifests,roots):
        for row in iter_manifest(path(manifest),root=path(root)):
            split=str(row.get(data['label_column']+'_split','')).strip()
            if split in ('','-1'):continue
            if split not in ('0','1','2'):raise ValueError('Expected CSV split 0=train, 1=test, 2=validation')
            label=parser._parse_label(row[data['label_column']],manifest,0)
            if not np.isfinite(label).all():raise ValueError('Nonfinite label')
            if parser.task=='classification' and not 0<=label<parser.num_class:raise ValueError('Invalid class label')
            row.update(_manifest=str(manifest),_label=label)
            splits.setdefault({'0':'train','1':'test','2':'validation'}[split],[]).append(row)
    return splits


def metrics_for(task,truth,logits,classes):
    if task=='classification':
        metrics,_,predictions,probabilities=classification_metrics(truth,logits,classes)
    elif task=='multi_label_classification':
        metrics,predictions,probabilities=multilabel_metrics(truth,logits,classes)
    else:
        predictions,probabilities=logits[:,0],None
        metrics=_regression_metrics(predictions,truth)
    return metrics,predictions,probabilities


def write_json(filename,value):
    def clean(item):
        if isinstance(item,dict):return {k:clean(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [clean(v) for v in item]
        if isinstance(item,(float,np.floating)) and not np.isfinite(item):return None
        return item.item() if isinstance(item,np.generic) else item
    filename.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False))


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
    parser.add_argument('--data-config', required=True, help='Native FAVOR task or data YAML; uses its original CSV splits')
    for mode in ('audio', 'video'):
        for split in ('validation', 'test'):
            parser.add_argument(f'--{mode}-{split}', required=split=='test')
        parser.add_argument(f'--{mode}-run-config', help='Saved run config with measured parameter_counts')
    parser.add_argument('--experiment', default='favor-late-fusion')
    parser.add_argument('--audio-weight', type=float, default=.5, help='Fixed probability weight when the CSV has no validation split')
    parser.add_argument('--test',action='store_true',help='Check existing real predictions without writing results')
    parser.add_argument('--save-predictions',action='store_true')
    parser.add_argument('--output',help='Output directory; defaults to OUTPUT/ablations/favor-late-fusion/<experiment>')
    args = parser.parse_args()
    cfg=load_config(path(args.data_config))
    data=cfg['data']
    if is_long_media_task(data) and not data.get('num_sentence'):data['num_sentence']=8
    output=path(args.output) if args.output else ROOT/'OUTPUT/ablations/favor-late-fusion'/args.experiment
    if not args.test:output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('favor.late_fusion')
    logging.basicConfig(level=logging.INFO)
    splits=paired_splits(data)
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
    if args.save_predictions and not args.test:
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
    if not args.test:
        (output/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
        write_json(output/'metrics.json', dict(best=best, test=metrics, samples=len(keys), source_parameter_counts=counts))
    logger.info('selection=%s audio_weight=%.2f test_instances=%d', best['selection'], best['audio_weight'], len(keys))


if __name__ == '__main__':
    main()
