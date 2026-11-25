"""CSV and restore regressions. Real-model integration is BASELINES.run --test."""
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
    save_checkpoint(tmp_path/'best.pt', model, optimizer, scheduler, scaler, loader, cfg, 1, 1, None, [], validation_loader=validation)
    draws = (random.random(), float(np.random.rand()), torch.rand(3))
    update(model(x).square().mean(), model, optimizer, scheduler, scaler)
    expected = copy.deepcopy(model.state_dict())
    resumed, opt, sched, scale, data = build()
    torch.rand(3, generator=validation.generator)
    state = restore(tmp_path/'best.pt', resumed, opt, sched, scale, data, cfg, validation)
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
        restore(tmp_path/'best.pt', resumed, opt, sched, scale, data, dict(cfg, max_batches=1))


def test_real_task_schema_allows_regression_without_class_count(tmp_path):
    import wave
    from BASELINES.common.data import read_rows
    with wave.open(str(tmp_path / 'a.wav'), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(bytes(3200))
    manifest = tmp_path/'regression.csv'
    manifest.write_text('audio_path,PHQ,PHQ_split,participant\na.wav,12.5,0,P01\n')
    cfg = dict(datasets=[str(manifest)], rootpaths=[str(tmp_path)], task='regression',
               label_column='PHQ', cohort='modality')
    rows, missing = read_rows(cfg, 'audio')
    assert missing == 0 and rows[0]['_label'] == 12.5


@pytest.mark.parametrize('stage,filename',[('pretrain','latest.pt'),('finetune','best.pt')])
def test_no_validation_uses_final_epoch_without_reading_test(tmp_path,stage,filename):
    import logging
    from BASELINES.common.runtime import fit, load_file
    cfg = dict(model={'kind':'test'}, data=dict(task='classification'),
               optimization=dict(lr=.01,epochs=2), stage=stage, seed=42)
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
    archives = ['epoch-0001.pt', 'epoch-0002.pt'] if stage == 'pretrain' else []
    assert sorted(p.name for p in tmp_path.glob('*.pt')) == [*archives, filename]
    best=load_file(tmp_path/filename)
    assert {'optimizer','scheduler','scaler','rng','loader_rng'}<=best.keys()
    assert calls==[0,1] and best['epoch']==2
    assert best['best']['score'] is None and best['best']['validation'] is None
    assert best['best']['selection']=='final_epoch_no_csv_validation'


def test_pretrain_archives_weights_and_logs_each_update_after_resume(tmp_path, monkeypatch):
    import csv
    from types import SimpleNamespace
    from BASELINES.common.runtime import load_file
    from BASELINES.pretrain import trainer

    class ToyEncoder(nn.Module):
        input_spec = {'modality': 'audio'}

        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.encoder.weight, .5)
            self.batch_losses = []

        def prepare(self, clip):
            return clip

        def pretrain_loss(self, batch, step):
            loss = self.encoder(torch.stack([clip['audio'] for clip in batch])).square().mean()
            self.batch_losses.append(float(loss.detach()))
            return loss

        def export_encoder(self):
            return self.encoder.state_dict()

    samples = [dict(sentences=[[dict(audio=torch.tensor([float(i)]))]]) for i in range(1, 8)]
    def make_loaders(cfg, spec, pretrain):
        loader = DataLoader(samples, batch_size=2, collate_fn=list,
                            generator=torch.Generator().manual_seed(42))
        return {'train': loader}

    monkeypatch.setattr(trainer, 'make_loaders', make_loaders)
    monkeypatch.setattr(trainer, 'STEP_LOSS_FLUSH_INTERVAL', 1)
    cfg = dict(stage='pretrain', route='scratch_pretrain', data={}, max_batches=4,
               model=dict(id='toy'), output=str(tmp_path), device='cpu',
               optimization=dict(lr=.01, epochs=2, gradient_accumulation=2), stop_after_epoch=1)
    model = ToyEncoder()
    args = SimpleNamespace(resume=None, evaluate=None)
    trainer.main(cfg, args, model=model)
    loss_file = tmp_path / 'step_loss.csv'
    with loss_file.open('a', newline='') as stream:
        csv.writer(stream).writerow((2, 3, 99))

    latest = tmp_path / 'latest.pt'
    cfg['stop_after_epoch'] = 2
    args.resume = latest
    trainer.main(cfg, args, model=model)
    with loss_file.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    assert [(int(row['epoch']), int(row['step'])) for row in records] == [(1, 1), (1, 2), (2, 3), (2, 4)]
    losses = model.batch_losses
    assert [float(row['loss']) for row in records] == pytest.approx([
        (losses[0] + losses[1]) / 2, (2 * losses[2] + losses[3]) / 3,
        (losses[4] + losses[5]) / 2, (2 * losses[6] + losses[7]) / 3,
    ])
    for number in (1, 2):
        archived = load_file(tmp_path / f'epoch-{number:04d}.pt')
        assert archived['epoch'] == number and archived['step'] == 2 * number
        assert 'model' in archived and 'encoder' in archived and 'optimizer' not in archived
    assert load_file(latest)['epoch'] == 2 and 'optimizer' in load_file(latest)


def test_wavjepa_prepares_a_short_clip_without_counting_unpadded_tokens():
    from BASELINES.build_model.audio import WavJEPA

    model = WavJEPA.__new__(WavJEPA)
    nn.Module.__init__(model)
    model.pretraining = True
    model.processor = lambda clips, **_: {'input_values': torch.from_numpy(clips[0])[None]}

    result = model.prepare({'audio': torch.arange(16, dtype=torch.float32)})
    assert result['audio'].shape == (1, 1, 32160)
    assert torch.isfinite(result['audio']).all()


def test_pretraining_replaces_and_remembers_empty_audio(tmp_path, caplog, monkeypatch):
    import csv
    import wave
    from BASELINES.common.data import PretrainDataset
    from BASELINES.common import pretrain_media_errors

    monkeypatch.setattr(pretrain_media_errors, 'OUTPUT_ROOT', tmp_path)
    empty, good = tmp_path / 'empty.wav', tmp_path / 'good.wav'
    for filename, count in ((empty, 0), (good, 1600)):
        with wave.open(str(filename), 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            audio.writeframes(bytes(2 * count))
    manifest = tmp_path / 'audio.csv'
    manifest.write_text(f'{empty} 0\n{good} 0\n')
    cfg = {'datasets': [str(manifest)]}
    spec = {'modality': 'audio', 'seconds': 0.1, 'sample_rate': 16000}
    dataset = PretrainDataset(cfg, spec)
    peer = PretrainDataset(cfg, spec)
    assert len(dataset) == 2
    assert dataset[0]['id'] == str(good)
    assert 'Skipping pretraining row 0' in caplog.text
    peer.media_errors.known(refresh=True)
    assert peer[0]['id'] == str(good)
    assert caplog.text.count('Skipping pretraining row 0') == 1
    reloaded = PretrainDataset(cfg, spec)
    assert len(reloaded) == 1 and reloaded[0]['id'] == str(good)
    with (tmp_path / 'pretrain/error_media_a.csv').open(newline='') as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1 and records[0]['name'] == str(empty)
    assert 'Empty media' in records[0]['reason']
    for modality in ('v', 'va'):
        assert (tmp_path / f'pretrain/error_media_{modality}.csv').read_text() == 'name,reason\n'


def test_pretraining_excludes_registered_video_pairs_by_id(tmp_path, monkeypatch):
    import csv
    from BASELINES.common.data import PretrainDataset
    from BASELINES.common import pretrain_media_errors

    monkeypatch.setattr(pretrain_media_errors, 'OUTPUT_ROOT', tmp_path)
    media_errors = pretrain_media_errors.PretrainMediaErrors('audio-video')
    for name in ('drop1', 'drop2', 'drop3'):
        media_errors.record(name, 'invalid pair')
    manifest = tmp_path / 'paired.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pair_id', 'source_id', 'audio_path', 'video_path'])
        for name in ('drop1', 'drop2', 'keep1', 'drop3', 'keep2'):
            writer.writerow([name, 'subject', 'subject.wav', 'subject.mp4'])
    dataset = PretrainDataset({'datasets': [str(manifest)]}, {'modality': 'audio-video'})
    monkeypatch.setattr(dataset, '_decode', lambda row: [row['pair_id']])
    assert len(dataset) == 2
    assert [dataset[i]['sentences'][0][0] for i in range(2)] == ['keep1', 'keep2']


@pytest.mark.parametrize('codes,attempts,failed', [
    ((75, 0), 2, False),
    ((75, 75), 2, True),
    ((1,), 1, True),
])
def test_matrix_retries_only_nonfinite_evaluation_once(tmp_path, monkeypatch, codes, attempts, failed):
    from types import SimpleNamespace
    from BASELINES.common import launch

    monkeypatch.setattr(launch, 'task_catalog', lambda: {})
    monkeypatch.setattr(launch, 'load_config', lambda _: {})
    monkeypatch.setattr(launch, 'resource_errors', lambda *_: [])
    calls = []
    def run(command, gpus, test):
        calls.append(command)
        return codes[len(calls) - 1]
    monkeypatch.setattr(launch, 'run_ranks', run)
    args = SimpleNamespace(stage='finetune', tasks=['ordinary'], gpus=2, test=False, set=[],
                           resume=None, evaluate=None, checkpoint=None, smoke=False)
    configs = [tmp_path / 'toy.yaml']
    if failed:
        with pytest.raises(SystemExit, match='1'):
            launch.matrix(configs, args)
    else:
        launch.matrix(configs, args)
    assert len(calls) == attempts and all(call == calls[0] for call in calls)


@pytest.mark.parametrize('task,metric,scores', [
    ('classification', 'f1_macro', (.6, .5, .7)),
    ('regression', 'mae', (2., 3., 1.)),
])
def test_mental_test_selection_keeps_best_and_resumes_latest(tmp_path, task, metric, scores):
    import logging
    from BASELINES.common.runtime import fit, load_file

    cfg = dict(model={'kind': 'test'}, data=dict(task=task),
               optimization=dict(lr=.01, epochs=3), stage='finetune', seed=42,
               stop_after_epoch=1)
    model = nn.Linear(2, 2)
    loader_args = dict(batch_size=2, generator=torch.Generator().manual_seed(42))
    train = DataLoader(TensorDataset(torch.ones(2, 2)), **loader_args)
    test = DataLoader(TensorDataset(torch.ones(2, 2)), **loader_args)
    calls = []
    inference_epochs = []

    def epoch(loader, optimizer, scheduler, scaler, step):
        if loader is train:
            update(model(torch.ones(2, 2)).square().mean(), model, optimizer, scheduler, scaler)
            return dict(loss=1., updates=1, samples=2)
        assert loader is test and not model.training and not torch.is_grad_enabled()
        calls.append(step)
        return {metric: scores[step - 1], 'updates': 0}

    def run(resume=None):
        def test_epoch(loader, step, number):
            inference_epochs.append(number)
            return epoch(loader, None, None, None, step)

        return fit(model, dict(train=train, test=test), cfg, tmp_path,
                   logging.getLogger('test'), torch.device('cpu'), resume, epoch,
                   test_epoch_fn=test_epoch)

    best_path, latest_path = tmp_path / 'best.pt', tmp_path / 'latest.pt'
    run()
    assert not calls and not latest_path.exists()
    cfg.update(task_name='mental/test-task', stop_after_epoch=2)
    run(best_path)
    assert calls == [1, 2]
    assert load_file(best_path)['epoch'] == 1
    latest = load_file(latest_path)
    assert latest['epoch'] == 2 and 'optimizer' in latest
    assert latest['best']['test'][metric] == scores[0]
    assert 'optimizer' not in load_file(best_path)
    cfg['stop_after_epoch'] = 3
    run(latest_path)
    assert calls == [1, 2, 3] and load_file(best_path)['epoch'] == 3

    best_path.unlink()
    run(latest_path)
    assert calls == [1, 2, 3] and load_file(best_path)['epoch'] == 3
    assert inference_epochs == [1, 2, 3]


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
        def prepare(self,clip):return {'x':clip['x'][None]}
        def forward(self,batch):return self.projection(batch['x'])
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

    infer = tmp_path / 'infer.csv'
    model.eval()
    with torch.no_grad():
        for number in (1, 2, 2):
            run_epoch(model, loader, nn.CrossEntropyLoss(), cfg, torch.device('cpu'),
                      predictions_path=infer, prediction_epoch=number)
    with infer.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['epoch', *predictions[0]]
        records = list(reader)
    assert [row['epoch'] for row in records] == ['1', '1', '2', '2']
    assert len({(row['epoch'], row['path']) for row in records}) == 4


def test_beats_batches_variable_length_waveforms_with_padding_mask():
    from BASELINES.build_model import TaskModel
    from BASELINES.build_model.audio import BEATsEncoder

    calls = []

    class DummyBEATs(nn.Module):
        def extract_features(self, source, padding_mask=None):
            calls.append((tuple(source.shape), None if padding_mask is None else padding_mask.clone()))
            tokens = torch.ones((source.shape[0], 4, 8), dtype=torch.float32)
            mask = None if padding_mask is None else torch.zeros((source.shape[0], 4), dtype=torch.bool)
            return tokens, mask

    adapter = BEATsEncoder.__new__(BEATsEncoder)
    nn.Module.__init__(adapter)
    adapter.encoder = DummyBEATs()
    adapter.output_dim = 8
    adapter.minimum_samples = 2800
    adapter.input_spec = dict(modality='audio', sample_rate=16000, seconds=10, minimum_samples=2800)

    model = TaskModel(adapter, 2, encoder_batch_size=8)
    samples = [
        dict(sentences=[[dict(audio=torch.ones(3000))]], windows=[[0]]),
        dict(sentences=[[dict(audio=torch.ones(5000))]], windows=[[0]]),
        dict(sentences=[[dict(audio=torch.ones(1000))]], windows=[[0]]),
    ]
    outputs = model.predict_samples(samples, 'classification', torch.device('cpu'))
    assert len(outputs) == 3
    assert len(calls) == 1
    shape, mask = calls[0]
    assert shape == (3, 5000)
    assert mask is not None and mask.shape == (3, 5000)
    assert (~mask[0, :3000]).all() and mask[0, 3000:].all()
    assert (~mask[1]).all()
    assert (~mask[2, :2800]).all() and mask[2, 2800:].all()

    same = adapter.batch_inputs([adapter.prepare(dict(audio=torch.ones(3200))) for _ in range(2)])
    assert same['source'].shape == (2, 3200) and 'padding_mask' not in same


def test_leavjepa_prepare_views_avoids_redundant_video_and_audio_transforms(monkeypatch):
    from torchvision.transforms import v2
    from BASELINES.build_model import audiovisual

    model = audiovisual.LeAVJEPA.__new__(audiovisual.LeAVJEPA)
    nn.Module.__init__(model)
    model.pretraining = True
    spec_batches = []

    class DummySpec(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('buf', torch.zeros(1))

        def forward(self, wav):
            spec_batches.append(tuple(wav.shape))
            return torch.zeros((wav.shape[0], 128, 801), device=wav.device)

    counts = dict(global_calls=0, local_calls=0)
    base_global = v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                              v2.ToDtype(torch.float32, scale=True)])
    base_local = v2.Compose([v2.RandomResizedCrop(224, scale=(.4, 1.), antialias=True),
                             v2.ToDtype(torch.float32, scale=True)])

    def count_global(frames):
        counts['global_calls'] += 1
        return base_global(frames)

    def count_local(frames):
        counts['local_calls'] += 1
        return base_local(frames)

    model.spectrogram = DummySpec()
    model.global_transform = count_global
    model.local_transform = count_local
    monkeypatch.setattr(audiovisual, 'frames_tensor',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('frames_tensor unused in pretrain')))

    clips = [dict(audio=torch.zeros(16000), video=torch.zeros((8, 64, 64, 3), dtype=torch.uint8)) for _ in range(4)]
    views = model.prepare_views(clips)
    assert spec_batches == [(3, 128000)]
    assert counts == dict(global_calls=2, local_calls=1)
    assert views['global_video'].shape == (1, 2, 3, 16, 224, 224)
    assert views['global_audio'].shape == (1, 2, 1, 128, 801)
    assert views['local_audio'].shape == (1, 1, 128, 801)
    assert views['local_video'].shape == (1, 3, 16, 224, 224)


def test_compile_model_is_idempotent_and_respects_disable_flags(monkeypatch):
    from BASELINES.common.distributed import compile_model

    calls = []

    class Dummy(nn.Module):
        def compile_modules(self):
            calls.append(1)

    m = Dummy()
    monkeypatch.setenv('TORCHDYNAMO_DISABLE', '1')
    compile_model(m)
    assert calls == []

    monkeypatch.delenv('TORCHDYNAMO_DISABLE', raising=False)
    monkeypatch.setenv('BASELINES_COMPILE', '0')
    compile_model(m)
    assert calls == []

    monkeypatch.setenv('BASELINES_COMPILE', '1')
    compile_model(m)
    compile_model(m)
    assert calls == [1]


def test_selected_tasks_supports_mental_core():
    from BASELINES.common.experiments import task_catalog
    from BASELINES.worker import selected_tasks

    expected = {
        'mental/RESULT-total',
        'mental/RESULT-Depression',
        'mental/RESULT-Anxiety',
        'mental/SCL-Overall',
        'mental/DSMxSCL-Depression',
        'mental/DSMxSCL-Anxiety',
        'mental/DSMxSCL-SomaticSymptoms',
        'mental/DSMxSCL-ObsessiveCompulsive',
        'mental/DSMxSCL-AngerHostility',
        'mental/DSMxSCL-PersonalityInterpersonal',
        'mental/DSM-SleepProblems',
        'mental/DSM-SuicidalIdeation',
        'mental/MINI',
    }
    chosen = selected_tasks(['mental_core'])
    assert len(chosen) == 13
    assert set(chosen) == expected
    catalog = task_catalog()
    assert all(name in catalog for name in chosen)
    assert selected_tasks(['mental_core', 'mental/MINI']) == chosen
