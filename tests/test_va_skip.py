import csv
import json
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from datasets.va_lejepa import VALeJEPADataset, collate_va_lejepa, synchronize_training_batch
from models.video_lejepa.sigreg import SIGReg
from test_va_lejepa import tiny_va, view


def broken_sample(index=0):
    return {'decode_error': dict(pair_id=f'bad{index}', source_id=f'bad{index}',
        video_path=f'/bad{index}.mp4', audio_path=f'/bad{index}.wav', reason='ValueError: sync mismatch')}


def good_sample(index=0):
    global_, local = view(1), view(1, 2, 1.)
    return {'global': {key: value[0] for key, value in global_.items()},
        'local': [{key: value[0] for key, value in local.items()}],
        'pair_id': str(index), 'source_id': str(index), 'video_path': 'v', 'audio_path': 'a',
        'sync_diagnostics': dict(duration_delta=0., start_delta=0., repeated=False, padded=False)}


class Samples(Dataset):
    epoch = 0
    def __init__(self, all_bad=False):
        self.all_bad = all_bad
    def __len__(self):
        return 5
    def __getitem__(self, index):
        return broken_sample(index) if self.all_bad or index % 2 else good_sample(index)


def test_dataset_skip_keeps_pair_identity_and_error_mode(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('pair_id,source_id,video_path,audio_path\nx,x,/a/x.mp4,/b/x.wav\n')
    calls = []
    class FailedDecoder:
        def __init__(self, row, *args):
            calls.append(row['pair_id'])
            raise ValueError('sync mismatch')
    cfg = dict(manifests=[str(manifest)], on_decode_error='skip')
    sample = VALeJEPADataset(cfg, decoder_factory=FailedDecoder)[0]
    assert sample['decode_error']['pair_id'] == 'x'
    assert sample['decode_error']['video_path'] == '/a/x.mp4'
    assert 'sync mismatch' in sample['decode_error']['reason']
    assert calls == ['x']  # no random replacement pair
    with pytest.raises(RuntimeError, match='pair_id=x'):
        VALeJEPADataset(dict(cfg, on_decode_error='error'), decoder_factory=FailedDecoder)[0]


def test_collate_filters_invalid_pairs_and_handles_empty_batch():
    batch = collate_va_lejepa([good_sample(), broken_sample(), good_sample(2)])
    assert batch['pair_id'] == ['0', '2']
    assert batch['global']['video'].shape[0] == 2
    assert batch['local']['video'].shape[0] == 2
    assert len(batch['skipped_samples']) == 1
    empty = collate_va_lejepa([broken_sample()])
    assert synchronize_training_batch(empty, torch.device('cpu')) == (None, 0)


class Sampler:
    def set_epoch(self, epoch):
        pass


def entry_config(folder):
    return dict(folder=str(folder), model={}, data={},
        optimization=dict(total_steps=2, grad_accum_steps=2, lr_new=.001),
        meta=dict(auto_resume=True, log_every_steps=1, save_every_steps=1),
        ema=dict(enabled=False), loss={'sigreg': {'num_proj': 4, 'knots': 3}})


def test_pretrain_skip_accumulation_logging_and_resume(tmp_path, monkeypatch):
    import app.pretrain_va_lejepa.train as train
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(train, 'build_va_lejepa', lambda *args, **kwargs: tiny_va())
    monkeypatch.setattr(train, 'make_va_lejepa_loader', lambda *args: (
        None, DataLoader(Samples(), batch_size=1, collate_fn=collate_va_lejepa), Sampler()))
    cfg = entry_config(tmp_path)
    assert train.main(cfg)['step'] == 2
    cfg['optimization']['total_steps'] = 3
    assert train.main(cfg)['step'] == 3
    state = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert state['step'] == 3
    with (tmp_path / 'logs/history.csv').open() as handle:
        assert [int(row['step']) for row in csv.DictReader(handle)] == [1, 2, 3]
    reports = [json.loads(line) for line in (tmp_path / 'logs/skipped_pairs_rank0.jsonl').read_text().splitlines()]
    assert {row['pair_id'] for row in reports} == {'bad1', 'bad3'}
    with (tmp_path / 'logs/data_skips.csv').open() as handle:
        summaries = list(csv.DictReader(handle))
    assert int(summaries[1]['skipped_batches']) == 2


def test_all_invalid_data_stops_instead_of_looping_forever(tmp_path, monkeypatch):
    import app.pretrain_va_lejepa.train as train
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(train, 'build_va_lejepa', lambda *args, **kwargs: tiny_va())
    monkeypatch.setattr(train, 'make_va_lejepa_loader', lambda *args: (
        None, DataLoader(Samples(all_bad=True), batch_size=1, collate_fn=collate_va_lejepa), Sampler()))
    with pytest.raises(RuntimeError, match='no usable VA training batches'):
        train.main(entry_config(tmp_path))


def _ddp_worker(rank, init_file, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        model = DDP(torch.nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        criterion = SIGReg(knots=3, num_proj=2)
        counts = [2, 0, 2] if rank == 0 else [1, 2, 2]
        results, discarded, micro = [], [], 0
        for count in counts:
            batch = {'global': {'video': torch.ones(count, 1)},
                     'local': {'video': torch.ones(count, 1, 1)},
                     'pair_id': [str(i) for i in range(count)], 'skipped_samples': []}
            batch, dropped = synchronize_training_batch(batch, torch.device('cpu'))
            discarded.append(dropped)
            results.append(None if batch is None else batch['global']['video'].shape[0])
            if batch is None:
                continue
            assert len(batch['pair_id']) == batch['local']['video'].shape[0]
            with model.no_sync() if micro == 0 else nullcontext():
                output = model(batch['global']['video'])
                loss = (criterion(output[None]) + output.square().mean()) / 2
                loss.backward()
            micro += 1
        assert micro == 2 and all(p.grad.isfinite().all() for p in model.parameters())
        optimizer.step()
        from pathlib import Path
        Path(result_dir, f'{rank}.json').write_text(json.dumps(dict(counts=results, discarded=discarded)))
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_skips_together_and_preserves_accumulation(tmp_path):
    mp.spawn(_ddp_worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / f'{rank}.json').read_text())
        assert result['counts'] == [1, None, 2]
        assert result['discarded'] == ([1, 0, 0] if rank == 0 else [0, 2, 0])
