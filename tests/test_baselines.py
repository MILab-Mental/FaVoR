"""Protocol/restore regressions. Real-model integration is BASELINES.run --smoke."""
from collections import OrderedDict
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BASELINES.common.data import partition
from BASELINES.common.runtime import (optimizer_and_scheduler, save_checkpoint, restore,
                                     strict_load, update, evaluation_rng)
from BASELINES.finetune.fusion import combine


def test_csv_assignments_are_preserved_without_group_resplitting():
    rows = [dict(_group='same-speaker', _split=v, id=i) for i,v in enumerate((0,1,0,2,1))]
    splits = partition(rows)
    assert [r['id'] for r in splits['train']] == [0,2]
    assert [r['id'] for r in splits['test']] == [1,4]
    assert [r['id'] for r in splits['validation']] == [3]
    assert 'validation' not in partition([dict(_split=0),dict(_split=1)])


def test_checkpoint_loading_rejects_unexplained_mismatch():
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match='incompatible'):
        strict_load(model, dict(weight=torch.zeros(3, 4), bias=torch.zeros(2)))
    with pytest.raises(ValueError, match='incompatible'):
        strict_load(model, dict(weight=torch.zeros(2, 4)))


def test_resume_reproduces_next_update_and_random_draws(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    cfg = dict(model={'kind': 'test'}, data={}, optimization=dict(lr=.01, epochs=2), seed=42, stage='finetune')
    def build():
        model = nn.Sequential(OrderedDict(encoder=nn.Linear(4, 4), dropout=nn.Dropout(.3), head=nn.Linear(4, 2)))
        optimizer, scheduler = optimizer_and_scheduler(model, cfg, steps=4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        loader = DataLoader(TensorDataset(torch.ones(2, 4)), generator=torch.Generator().manual_seed(42))
        return model, optimizer, scheduler, scaler, loader
    model, optimizer, scheduler, scaler, loader = build()
    validation = DataLoader(TensorDataset(torch.ones(2, 4)), generator=torch.Generator().manual_seed(43))
    torch.rand(3, generator=validation.generator)
    validation_state = validation.generator.get_state()
    training_rng = torch.get_rng_state()
    with evaluation_rng(validation):
        evaluation_draw = torch.rand(3)
        torch.rand(3, generator=validation.generator)
    assert torch.equal(torch.get_rng_state(), training_rng)
    assert torch.equal(validation.generator.get_state(), validation_state)
    with evaluation_rng(validation):
        assert torch.equal(torch.rand(3), evaluation_draw)
    x = torch.arange(8).float().reshape(2, 4)
    update(model(x).square().mean(), model, optimizer, scheduler, scaler)
    save_checkpoint(tmp_path/'last.pt', model, optimizer, scheduler, scaler, loader, cfg, 1, 1, None, [], validation_loader=validation)
    draws = (random.random(), float(np.random.rand()), torch.rand(3))
    update(model(x).square().mean(), model, optimizer, scheduler, scaler)
    expected = copy.deepcopy(model.state_dict())
    resumed, opt, sched, scale, data = build()
    torch.rand(3, generator=validation.generator)
    state = restore(tmp_path/'last.pt', resumed, opt, sched, scale, data, cfg, validation)
    assert torch.equal(validation.generator.get_state(), validation_state)
    assert state['step'] == 1 and sched.last_epoch == 1
    assert random.random() == draws[0]
    assert float(np.random.rand()) == draws[1]
    assert torch.equal(torch.rand(3), draws[2])
    update(resumed(x).square().mean(), resumed, opt, sched, scale)
    assert all(torch.equal(expected[k], v) for k, v in resumed.state_dict().items())
    assert resumed._gradient_check['encoder']['absolute_update'] > 0
    assert resumed._gradient_check['head']['absolute_update'] > 0
    with pytest.raises(ValueError, match='max_batches'):
        restore(tmp_path/'last.pt', resumed, opt, sched, scale, data, dict(cfg, max_batches=1))


def test_fusion_averages_probabilities_and_scalar_regression():
    a, v = np.array([[.9, .1]]), np.array([[.2, .8]])
    assert np.allclose(np.exp(combine(a, v, .25, 'classification')), .25*a+.75*v)
    assert np.allclose(combine(np.array([1.]), np.array([3.]), .5, 'regression'), [[2.]])


def test_real_task_schema_allows_regression_without_class_count(tmp_path):
    from BASELINES.common.data import read_rows
    manifest = tmp_path/'regression.csv'
    manifest.write_text('audio_path,PHQ,PHQ_split,participant\na.wav,12.5,0,P01\n')
    cfg = dict(datasets=[str(manifest)], rootpaths=[str(tmp_path)], task='regression',
               label_column='PHQ', cohort='modality')
    rows, missing = read_rows(cfg, 'audio')
    assert missing == 0 and rows[0]['_label'] == 12.5


def test_native_branch_supervision_reaches_auxiliary_head():
    from BASELINES.finetune.trainer import run_epoch
    from app.finetune_va.train import _loss
    class NativeBranches(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.head, self.aux = nn.Linear(2, 4), nn.Linear(4, 2), nn.Linear(4, 2)
        def predict_sample(self, sample, task, device):
            x = self.encoder(sample['x'][None])
            return dict(logits=self.head(x), branches=dict(aux=self.aux(x)))
        def supervised_loss(self, output, labels, criterion, task):
            return _loss(output, labels, criterion, task, .1)
    model = NativeBranches()
    before = model.aux.weight.detach().clone()
    dataset = [dict(x=torch.tensor([1., float(i)]), label=i, id=str(i)) for i in range(2)]
    loader = DataLoader(dataset, batch_size=2, collate_fn=list)
    cfg = dict(data=dict(task='classification', num_class=2), optimization=dict(lr=.01))
    optimizer, scheduler = optimizer_and_scheduler(model, cfg, 2)
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), cfg, torch.device('cpu'),
                        optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False))
    assert metrics['samples'] == 2 and metrics['updates'] == 1
    assert not torch.equal(before, model.aux.weight)


def test_no_validation_uses_final_epoch_without_reading_test(tmp_path):
    import logging
    from BASELINES.common.runtime import fit, load_file
    cfg = dict(model={'kind':'test'}, data=dict(task='classification'),
               optimization=dict(lr=.01,epochs=2), stage='finetune', seed=42)
    model = nn.Sequential(OrderedDict(encoder=nn.Linear(2,2),head=nn.Linear(2,2)))
    train = DataLoader(TensorDataset(torch.ones(2,2)), batch_size=2,
                       generator=torch.Generator().manual_seed(42))
    calls = []
    def epoch(loader, optimizer, scheduler, scaler, step):
        assert loader is train and optimizer is not None
        calls.append(step)
        loss = model(torch.ones(2,2)).square().mean()
        update(loss,model,optimizer,scheduler,scaler)
        return dict(loss=float(step+1),updates=1,samples=2)
    fit(model,dict(train=train,test=object()),cfg,tmp_path,logging.getLogger('test'),
        torch.device('cpu'),None,epoch)
    best=load_file(tmp_path/'best.pt')
    assert calls==[0,1] and best['epoch']==2
    assert best['best']['score'] is None and best['best']['validation'] is None
    assert best['best']['selection']=='final_epoch_no_csv_validation'


def test_mental_instances_match_favor_and_train_attention_before_classification(tmp_path):
    from types import SimpleNamespace
    from datasets.common.instances import SentenceInstanceDataset
    from BASELINES.common.data import configure_instances, MediaDataset
    from BASELINES.build_model import TaskModel
    from BASELINES.finetune.trainer import run_epoch
    cfg = dict(data=dict(datasets=['mental_LV.csv'],task='classification',num_class=2,
                         num_sentence=2,sentence_eval_stride=1),
               optimization=dict(lr=.01),model={})
    configure_instances(cfg,'video')
    rows = [dict(video_path=f'/media/{who}-s{number:04}.mp4',_label=label)
            for who,number,label in [('p1',3,0),('p1',1,0),('p1',2,0),('p2',1,1),('p2',2,1)]]
    dataset = MediaDataset(rows,cfg['data'],dict(modality='video'),False)
    native = SentenceInstanceDataset(SimpleNamespace(samples=[r['video_path'] for r in rows],
        labels=[r['_label'] for r in rows]),2,'video',False,1)
    assert dataset.samples==native.samples
    assert [[r['video_path'] for r in group] for group in dataset.rows]==[
        [rows[i]['video_path'] for i in group] for group in native.groups]
    def decode(row):
        number=int(Path(row['video_path']).stem.rsplit('-s',1)[1])
        return [dict(x=torch.tensor([float(number),float(number**2),float(row['_label'])]))]
    from pathlib import Path
    dataset._decode=decode
    class Encoder(nn.Module):
        output_dim=4
        input_spec=dict(modality='video')
        def __init__(self):
            super().__init__()
            self.projection=nn.Linear(3,4)
        def prepare(self,clip):return clip
        def forward(self,batch):return self.projection(batch['x'][None])
    model=TaskModel(Encoder(),2,cfg['model']['instance_aggregator'])
    before=model.head.instance_aggregator.attention[0].weight.detach().clone()
    loader=DataLoader(dataset,batch_size=2,collate_fn=list)
    optimizer,scheduler=optimizer_and_scheduler(model,cfg,2)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    train=run_epoch(model,loader,nn.CrossEntropyLoss(),cfg,torch.device('cpu'),optimizer,scheduler,scaler)
    assert train['samples']==2 and train['updates']==1 and np.isfinite(train['loss'])
    assert not torch.equal(before,model.head.instance_aggregator.attention[0].weight)
    with torch.no_grad():
        metrics=run_epoch(model,loader,nn.CrossEntropyLoss(),cfg,torch.device('cpu'),predictions_path=tmp_path/'predictions.csv')
    import csv
    with (tmp_path/'predictions.csv').open() as handle:predictions=list(csv.DictReader(handle))
    assert metrics['samples']==2 and len(predictions)==2
    assert all('-s' not in p['path'] for p in predictions)
