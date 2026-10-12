"""Instance ownership, sentence sampling, feature gradients and full training."""
import csv
import json
from datetime import timedelta
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from app.main import load_config
from datasets.common.instances import (
    SentenceInstanceDataset, as_sentence_instances, sentence_identity, sentence_windows,
)
from models.finetune_a_model import AudioFineTuneModel
from models.finetune_v_model import ClipEncoder
from models.instance_aggregator import InstanceAggregator, InstancePredictionHead
from models.va_lejepa.finetune import VAFinetuner
from models.video_jepa.vision_transformer import VisionTransformer
from utils.instance_evaluation import average_window_outputs, predict_instance_batch
from utils.instance_finetune_configs import is_long_media_task, select_instance_presets
from test_va_lejepa import tiny_audio, tiny_va, view


class Sentences(Dataset):
    def __init__(self, modality, task='classification'):
        self.modality, self.task, self.epoch = modality, task, 0
        self.samples = [f'/media/{name}-s{i:04d}.wav' for name, count in [('first', 10), ('second', 3)]
                        for i in range(1, count + 1)]
        labels = [0, 1] if task == 'classification' else [.1, .9] if task == 'regression' else [[1., 0.], [0., 1.]]
        self.labels = [labels[0]] * 10 + [labels[1]] * 3
        self.rows = [{'video_path': str(Path(p).with_suffix('.mp4')), 'audio_path': p} for p in self.samples]
        self.calls = []

    def __len__(self):
        return len(self.samples)

    def get_sentence(self, index):
        self.calls.append(index)
        generator = torch.Generator().manual_seed(index)
        label = torch.as_tensor(self.labels[index], dtype=torch.long if self.task == 'classification' else torch.float32)
        if self.modality == 'audio':
            return torch.randn(1, 200, generator=generator), label, self.samples[index]
        if self.modality == 'video':
            return [[torch.randn(3, 4, 16, 16, generator=generator)]], label, [np.arange(4)], self.rows[index]['video_path']
        global_ = {key: value[0] for key, value in view(1).items()}
        return {
            'global': global_, 'local': [], 'label': label, 'pair_id': str(index), 'source_id': str(index),
            **self.rows[index], 'sync_diagnostics': {}}


def tiny_video():
    backbone = VisionTransformer(img_size=16, patch_size=8, num_frames=4, tubelet_size=1,
                                 embed_dim=24, depth=2, num_heads=3, out_layers=[0, 1])
    return ClipEncoder(backbone, instance_enabled=True)


def instance_pair(modality, task='classification'):
    return tuple(SentenceInstanceDataset(Sentences(modality, task), 8, modality, training)
                 for training in (True, False))


def test_numeric_order_padding_and_sliding_tail():
    assert sentence_identity('/root/name.with.dot-s0012.wav') == ('/root/name.with.dot', 12)
    assert sentence_identity('/other/name.with.dot-s0012.wav')[0] != sentence_identity('/root/name.with.dot-s0012.wav')[0]
    assert sentence_windows([0, 1], 4) == [[0, 1, 1, 1]]
    assert sentence_windows(list(range(6)), 4) == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    assert sentence_windows(list(range(9)), 4, 3) == [[0, 1, 2, 3], [3, 4, 5, 6], [5, 6, 7, 8]]
    data = Sentences('audio')
    # Ordering follows numeric suffixes even when manifest rows are shuffled.
    data.samples[0], data.samples[9] = data.samples[9], data.samples[0]
    wrapped = SentenceInstanceDataset(data, 8, 'audio', False)
    assert wrapped.groups[0] == [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]
    assert as_sentence_instances(data, {}, 'audio', True) is data
    assert as_sentence_instances(data, {'num_sentence': 0}, 'audio', True) is data


@pytest.mark.parametrize('modality', ['audio', 'video', 'audio-video'])
def test_instance_batch_shape_selection_and_last_sentence_repetition(modality):
    train, validation = instance_pair(modality)
    assert len(train) == len(validation) == 2 and train.labels == [0, 1]
    random.seed(7)
    batch = next(iter(DataLoader(train, batch_size=2, collate_fn=train.collate_fn)))
    assert len(set(train.dataset.calls[:8])) == 8
    assert set(train.dataset.calls[:8]) <= set(range(10))
    if modality == 'audio':
        inputs, labels, paths = batch
        assert inputs.shape == (2, 8, 1, 200)
        assert torch.equal(inputs[1, 2], inputs[1, 7])
    elif modality == 'video':
        inputs, labels, _, paths = batch
        assert inputs[0][0].shape == (2, 8, 3, 4, 16, 16)
        assert torch.equal(inputs[0][0][1, 2], inputs[0][0][1, 7])
    else:
        inputs, labels, paths = batch['global'], batch['label'], batch['pair_id']
        assert inputs['video'].shape == (2, 8, 3, 4, 16, 16)
        assert torch.equal(inputs['video'][1, 2], inputs['video'][1, 7])
    assert labels.tolist() == [0, 1]
    assert len(paths) == 2
    samples = next(iter(DataLoader(validation, batch_size=2, collate_fn=validation.collate_fn)))['instance_samples']
    assert [len(s['windows']) for s in samples] == [3, 1]
    assert samples[0]['windows'] == [list(range(i, i + 8)) for i in range(3)]
    assert samples[1]['windows'][0] == [0, 1, 2, 2, 2, 2, 2, 2]
    assert validation.dataset.calls == list(range(13))  # no repeated decoding for overlapping windows


def test_sentence_decode_failure_never_changes_instance():
    data = Sentences('audio')
    def fail(index):
        raise RuntimeError(f'broken sentence {index}')
    data.get_sentence = fail
    with pytest.raises(RuntimeError, match='broken sentence'):
        SentenceInstanceDataset(data, 8, 'audio', True)[0]


@pytest.mark.parametrize('task', ['classification', 'regression', 'multi_label_classification'])
def test_average_predictions_then_one_record_per_instance(task):
    _, validation = instance_pair('audio', task)
    batch = next(iter(DataLoader(validation, batch_size=2, collate_fn=validation.collate_fn)))
    values = [torch.tensor([[4., -1.]]), torch.tensor([[-1., 2.]])]
    averaged = average_window_outputs(values, task)
    if task == 'classification':
        assert torch.allclose(averaged.softmax(-1), torch.stack([v.softmax(-1) for v in values]).mean(0))
    elif task == 'multi_label_classification':
        assert torch.allclose(averaged.sigmoid(), torch.stack([v.sigmoid() for v in values]).mean(0))
    else:
        assert torch.allclose(averaged, torch.tensor([[1.5, .5]]))
    calls = []
    def predict(window):
        calls.append(window[-1][0])
        return torch.tensor([[float(len(calls))]]) if task == 'regression' else values[(len(calls) - 1) % 2]
    outputs, labels, identities = predict_instance_batch(batch, predict, task)
    assert len(calls) == 4 and outputs.shape[0] == labels.shape[0] == len(identities) == 2
    if task == 'regression':
        assert outputs[:, 0].tolist() == [2., 4.]


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
@pytest.mark.parametrize('task', ['classification', 'regression', 'multi_label_classification'])
@pytest.mark.parametrize('modality', ['audio', 'video', 'audio-video'])
def test_instance_models_backpropagate_to_all_sentences_and_encoder_families(modality, task, pooling):
    cfg = {'pooling': pooling, 'hidden_dim': 8}
    outputs = 1 if task == 'regression' else 2
    if modality == 'audio':
        model = AudioFineTuneModel(dict(tiny_audio(), instance_aggregator=cfg), task, 2)
        inputs = torch.randn(2, 8, 1, 200, requires_grad=True)
        logits = model(inputs)
        feature_parameter = model.backbone.feature_extractor.conv_blocks[0].conv.weight
    elif modality == 'video':
        model = torch.nn.ModuleDict({'encoder': tiny_video(), 'head': InstancePredictionHead(24, outputs, cfg)})
        inputs = torch.randn(2, 8, 3, 4, 16, 16, requires_grad=True)
        logits = model['head'](model['encoder']([[inputs]])[0])
        feature_parameter = model['encoder'].backbone.patch_embed.proj.weight
    else:
        model = VAFinetuner(tiny_va(('feature', 'late')), outputs, {'instance_aggregator': cfg})
        inputs = torch.randn(2, 8, 3, 4, 16, 16, requires_grad=True)
        global_ = {key: value.reshape(2, 8, *value.shape[1:]) for key, value in view(16).items()}
        global_['video'] = inputs
        logits = model({'global': global_})['logits']
        feature_parameter = model.va.video_encoder.patch_embed.proj.weight
    assert logits.shape == (2, outputs)
    labels = torch.tensor([0, 1]) if task == 'classification' else torch.tensor([.1, .9]) if task == 'regression' else torch.eye(2)
    criterion = torch.nn.CrossEntropyLoss() if task == 'classification' else torch.nn.MSELoss() if task == 'regression' else torch.nn.BCEWithLogitsLoss()
    criterion(logits.squeeze(-1) if task == 'regression' else logits, labels).backward()
    assert (inputs.grad.flatten(2).abs().sum(-1) > 0).all()
    assert feature_parameter.grad is not None and feature_parameter.grad.abs().sum() > 0
    if pooling == 'attention':
        assert all(p.grad is not None for name, p in model.named_parameters() if 'instance_aggregator' in name and p.requires_grad)
    if modality == 'audio-video':
        assert model.va.audio_encoder.backbone.feature_extractor.conv_blocks[0].conv.weight.grad.abs().sum() > 0
        assert all(p.grad is not None for p in model.va.input_adapters.parameters())


def test_mean_aggregator_is_feature_mean_and_attention_handles_one_sentence():
    features = torch.randn(2, 8, 12)
    assert torch.equal(InstanceAggregator(12, {'pooling': 'mean'})(features), features.mean(1))
    aggregator = InstanceAggregator(12, {'pooling': 'attention'})
    assert torch.equal(aggregator(features[:, :1]), features[:, 0])


@pytest.mark.parametrize('modality', ['audio', 'video', 'audio-video'])
def test_one_sentence_input_with_instance_model(modality):
    cfg = {'pooling': 'attention', 'hidden_dim': 8}
    if modality == 'audio':
        model = AudioFineTuneModel(dict(tiny_audio(), instance_aggregator=cfg), 'classification', 2).eval()
        inputs = torch.randn(2, 1, 200)
        assert torch.allclose(model(inputs), model(inputs[:, None]))
    elif modality == 'video':
        model = tiny_video().eval()
        head = InstancePredictionHead(24, 2, cfg).eval()
        inputs = torch.randn(2, 3, 4, 16, 16)
        assert torch.allclose(head(model([[inputs]])[0]), head(model([[inputs[:, None]]])[0]))
    else:
        model = VAFinetuner(tiny_va(), 2, {'instance_aggregator': cfg}).eval()
        inputs = view(2)
        assert torch.allclose(model({'global': inputs})['logits'],
                              model({'global': {key: value[:, None] for key, value in inputs.items()}})['logits'])


@pytest.mark.parametrize('modality', ['audio', 'video', 'audio-video'])
def test_instance_loader_spawn_workers(modality):
    train, validation = instance_pair(modality)
    for dataset in (train, validation):
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn,
                            num_workers=1, multiprocessing_context='spawn')
        batch = next(iter(loader))
        if not dataset.training:
            assert len(batch['instance_samples']) == 2
        elif modality == 'audio-video':
            assert batch['global']['video'].shape[:2] == (2, 8)
        else:
            assert batch[1].tolist() == [0, 1]


def _ddp_instance_worker(rank, init_file, result_dir):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from app.finetune_a.train import DistributedEvaluationSampler
    from app.finetune_v.train import _task_outputs
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        results = {}
        for modality in ['audio', 'video', 'audio-video']:
            cfg = {'pooling': 'attention', 'hidden_dim': 8}
            train, validation = instance_pair(modality)
            if modality == 'video':
                encoder, head = DDP(tiny_video()), DDP(InstancePredictionHead(24, 2, cfg))
                parameters = list(encoder.parameters()) + list(head.parameters())
                forward = lambda batch: _task_outputs(encoder, head, batch[0], False)
                evaluate = lambda batch: _task_outputs(encoder.module, head.module, batch[0], False)
            else:
                model = DDP(AudioFineTuneModel(dict(tiny_audio(), instance_aggregator=cfg), 'classification', 2)
                            if modality == 'audio' else VAFinetuner(tiny_va(), 2, {'instance_aggregator': cfg}))
                parameters = list(model.parameters())
                forward = lambda batch: model(batch if modality == 'audio-video' else batch[0])
                evaluate = lambda batch: model.module(batch if modality == 'audio-video' else batch[0])
            optimizer = torch.optim.SGD(parameters, lr=.01)
            loader = DataLoader(train, batch_size=1, sampler=DistributedSampler(train, 2, rank), collate_fn=train.collate_fn)
            batch = next(iter(loader))
            output = forward(batch)
            logits = output['logits'] if isinstance(output, dict) else output
            labels = batch['label'] if modality == 'audio-video' else batch[1]
            torch.nn.functional.cross_entropy(logits, labels).backward()
            optimizer.step()
            validation_loader = DataLoader(validation, batch_size=1,
                sampler=DistributedEvaluationSampler(validation, rank, 2), collate_fn=validation.collate_fn)
            with torch.no_grad():
                evaluated, _, identities = predict_instance_batch(next(iter(validation_loader)), evaluate, 'classification')
            dist.barrier()  # ranks evaluated different window counts (3 vs 1)
            all_identities = [None, None]
            dist.all_gather_object(all_identities, identities)
            results[modality] = sorted(identity for shard in all_identities for identity in shard)
        Path(result_dir, f'rank{rank}.json').write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_two_rank_training_and_unequal_window_evaluation(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_instance_worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        results = json.loads((tmp_path / f'rank{rank}.json').read_text())
        for identities in results.values():
            assert identities == ['/media/first', '/media/second']


@pytest.mark.parametrize('modality', ['audio', 'video', 'audio-video'])
def test_training_entry_instance_reports_checkpoint_and_resume(tmp_path, monkeypatch, modality):
    import app.finetune_a.train as audio
    import app.finetune_v.train as video
    import app.finetune_va.train as va
    from models.va_lejepa import save_checkpoint
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    aggregator = {'pooling': 'attention', 'hidden_dim': 8}
    folder = tmp_path / modality
    config = dict(folder=str(folder), meta=dict(dtype='float32', frozen_encoder=False),
                  data=dict(task='classification', num_class=2, batch_size=2, num_workers=0, datasets=['unused'],
                            label_column='emotion', num_sentence=8, sample_rate=100, process_seconds=2,
                            crop_size=16, patch_size=8, tubelet_size=1, dataset_fpcs=[4]),
                  optimization=dict(epochs=1, lr=.001, head_lr=.001, backbone_lr=.001,
                                    warmup=0, start_lr=.001, final_lr=0, weight_decay=0, final_weight_decay=0),
                  model=dict(tiny_audio(), instance_aggregator=aggregator))
    if modality == 'audio':
        train = audio
        monkeypatch.setattr(train, 'make_audiodataset_finetune_a', lambda *a, **k: instance_pair('audio'))
        monkeypatch.setattr(train, 'load_audio_backbone', lambda *a, **k: None)
    elif modality == 'video':
        train = video
        config['model'] = dict(model_name='tiny', out_layers=[0, 1], instance_aggregator=aggregator)
        monkeypatch.setattr(train, 'make_videodataset_finetune_v', lambda *a, **k: instance_pair('video'))
        monkeypatch.setattr(train, 'build_model', lambda *a, **k: (tiny_video(), InstancePredictionHead(24, 2, aggregator)))
    else:
        train = va
        pretrained = tmp_path / 'va-pretrain.pt'
        save_checkpoint(pretrained, tiny_va())
        config['finetune'] = dict(pretrained_checkpoint=str(pretrained))
        monkeypatch.setattr(train, 'build_va_lejepa', lambda *a, **k: tiny_va())
        monkeypatch.setattr(train, 'AVCSVDataset', lambda cfg, split, training: Sentences('audio-video'))
    train.main(config)
    state = torch.load(folder / 'latest.pt', weights_only=False)
    assert (folder / 'best.pt').is_file()
    prediction = folder / 'logs' / ('predictions.csv' if modality == 'audio-video' else 'eval_best_predict.csv')
    with prediction.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # two instances, not thirteen sentences or four windows
    weights = state['model'] if modality == 'audio-video' else state['task_head']
    assert any('instance_aggregator' in key for key in weights)
    config['optimization']['epochs'] = 2
    train.main(config)
    resumed = torch.load(folder / 'latest.pt', weights_only=False)
    assert resumed['epoch'] == 2


def test_all_long_tasks_select_instance_presets_and_full_finetuning():
    counts = {}
    for path in Path('CONFIGS/tasks/finetune').rglob('*.yaml'):
        config = load_config(path)
        if is_long_media_task(config['data']):
            counts[config['app']] = counts.get(config['app'], 0) + 1
            assert config['data']['num_sentence'] == 8
            assert config['model']['instance_aggregator']['pooling'] in {'mean', 'attention'}
            assert config['folder'].endswith('/instance-sentences8')
            if config['app'] == 'finetune_va':
                assert not any(config['finetune'][key] for key in ['freeze_video_encoder', 'freeze_audio_encoder', 'freeze_fusion_adapter'])
            else:
                assert config['meta']['frozen_encoder'] is False
            updated = select_instance_presets(yaml.safe_load(path.read_text()), config['data'])
            assert updated['folder'] == config['folder']
        else:
            assert config['data'].get('num_sentence', 0) == 0
            assert 'instance_aggregator' not in config['model']
    assert counts == {'finetune_a': 44, 'finetune_v': 39, 'finetune_va': 39}
