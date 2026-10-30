"""Reuse the 33 ordinary and 38 mental task definitions; no generated configs."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

from app.main import load_config
from BASELINES.common.runtime import ROOT, path


@lru_cache(maxsize=1)
def task_catalog():
    tasks, seen = {}, set()
    for mode in ('audio', 'audio-video'):
        for group in ('classification', 'multilabel', 'regression'):
            for filename in sorted((ROOT / f'CONFIGS/data/finetune/{mode}/{group}').glob('*.yaml')):
                data = load_config(filename)['data']
                identity = (tuple(data['datasets']), data['label_column'], data['task'])
                if identity not in seen:
                    seen.add(identity)
                    tasks['ordinary/'+filename.stem] = data
    for filename in sorted((ROOT / 'CONFIGS/data/finetune/audio-video/mental').glob('*.yaml')):
        tasks['mental/'+filename.stem] = load_config(filename)['data']
    assert sum(k.startswith('ordinary/') for k in tasks)==33
    assert sum(k.startswith('mental/') for k in tasks)==38
    return tasks


def task_config(base, name, modality, test=False):
    cfg = deepcopy(base)
    cfg['stage']='finetune'
    cfg['data'].update(deepcopy(task_catalog()[name]))
    cfg['data'].pop('num_class', None)
    cfg['data'].update(deepcopy(task_catalog()[name]))
    cfg['data']['num_sentence'] = 0
    cfg['data'].pop('sentence_eval_stride', None)
    cfg['model'].pop('instance_aggregator', None)
    cfg.pop('instance_protocol', None)
    cfg.pop('output', None)
    paired=all(Path(p).stem.endswith(('_SV','_LV')) for p in cfg['data']['datasets'])
    cfg['data']['cohort'] = 'paired' if paired or modality!='audio' else 'modality'
    cfg['task_name'] = name
    route='direct' if not cfg['model'].get('encoder_checkpoint') else ('scratch-pretrained' if cfg['model']['kind'] in ('audiomae','leavjepa','avsiam_large','socialmae') else 'continued-pretrained')
    cfg['experiment'] = route+'/'+name
    cfg['no_save'] = test
    if test:
        cfg['data'].update(limit=4, batch_size=1, num_workers=0)
        cfg.update(max_batches=1, stop_after_epoch=1)
    return cfg


def resource_errors(cfg,test=False):
    model = cfg['model']
    kind = model['kind']
    issues = []
    def weights(spec):
        if not spec.get('weights'):return
        target=path(spec['weights'])
        if not target.exists():issues.append('missing weights: '+str(target))
        elif target.is_dir() and not any(p.stat().st_size>1024 for pattern in ('*.safetensors','*.bin') for p in target.glob(pattern)):
            issues.append('missing model tensors in snapshot: '+str(target))
    weights(model)
    if model.get('encoder_checkpoint') and not test and not path(model['encoder_checkpoint']).exists():
        issues.append('missing research-pretrained encoder: '+str(path(model['encoder_checkpoint'])))
    for branch in ('audio', 'video'):
        if isinstance(model.get(branch), dict):
            weights(model[branch])
    requirements = dict(eat=['fairseq'], avhubert=['fairseq','python_speech_features','dlib','skimage'],
                        languagebind=['peft','pytorchvideo'])
    packages=requirements.get(kind,[])
    interpreter=cfg.get('python',sys.executable)
    if interpreter!=sys.executable:
        if not Path(interpreter).is_file():
            issues.append('missing compatible Python environment: '+interpreter)
            return issues
        probe='import importlib.util,importlib.metadata,json,sys; print(json.dumps(dict(missing=[p for p in sys.argv[1:] if importlib.util.find_spec(p) is None],transformers=importlib.metadata.version("transformers"))))'
        result=subprocess.run([interpreter,'-B','-c',probe,*packages],capture_output=True,text=True)
        if result.returncode:
            issues.append('compatible environment is incomplete: '+result.stderr.strip().splitlines()[-1])
            return issues
        environment=json.loads(result.stdout)
    else:
        from importlib.metadata import version
        environment=dict(missing=[p for p in packages if importlib.util.find_spec(p) is None],transformers=version('transformers'))
    for package in environment['missing']:issues.append('missing dependency: '+package)
    if kind=='languagebind':
        if not environment['transformers'].startswith('4.30.'):
            issues.append('official LanguageBind needs its compatible transformers 4.30 environment; installed '+environment['transformers'])
    if kind=='avhubert':
        if cfg['data'].get('video_column')=='lip_video_path':
            import csv
            for manifest in cfg['data']['datasets']:
                with path(manifest).open(encoding='utf-8-sig',newline='') as handle:
                    columns=csv.DictReader(handle).fieldnames or []
                if 'lip_video_path' not in columns:issues.append(str(manifest)+': no lip_video_path ROI input')
        else:
            assets=model.get('mouth_preprocessing',{})
            for key in ('predictor','detector','mean_face'):
                if not assets.get(key) or not path(assets[key]).is_file():issues.append('missing mouth preprocessing asset: '+key)
    return issues
