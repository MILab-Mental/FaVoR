"""Exercise audio worker startup after distributed runtime initialization."""
import csv
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from app.finetune_a.train import _loader_kwargs


def test_loader_without_workers():
    kwargs = _loader_kwargs(dict(batch_size=2, num_workers=0, persistent_workers=True))
    assert not kwargs['persistent_workers']
    assert 'multiprocessing_context' not in kwargs
    assert 'timeout' not in kwargs


def _train_worker(rank, backend, init_file, folder, manifest):
    import app.finetune_a.train as train

    torch.set_num_threads(1)
    if backend == 'nccl':
        # Match app.main: each trainer sees one GPU under its cuda:0 index.
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.cuda.set_device(0)
    else:
        torch.cuda.is_available = lambda: False
    dist.init_process_group(backend, init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        cfg = dict(
            folder=folder, meta=dict(dtype='float32', frozen_encoder=False),
            data=dict(datasets=[manifest], rootpaths=[str(Path(manifest).parent)],
                      label_column='label', task='classification', num_class=2,
                      sample_rate=100, process_seconds=2, num_clips=1,
                      num_sentence=8, sentence_eval_stride=1, batch_size=1,
                      num_workers=1, persistent_workers=True, loader_timeout=60),
            optimization=dict(epochs=2, head_lr=.001, backbone_lr=.001, final_lr=0),
            model=dict(
                backbone=dict(mode='emotion2vec', conv_pos_depth=1, conv_pos_width=3,
                              conv_pos_groups=2, prenet_depth=1, num_extra_tokens=2),
                extractor=dict(conv_layers=[[8, 4, 2], [8, 2, 2]], mode='layer_norm'),
                encoder=dict(depth=1, d_model=8, nhead=2, dim_feedforward=16,
                             dropout=0., attention_dropout=0., ffn_targets=True),
                instance_aggregator=dict(pooling='attention', hidden_dim=8)),
        )
        # This fixture starts from random weights; checkpoint loading is
        # covered separately by the existing fine-tuning tests.
        train.load_audio_backbone = lambda *a, **k: None
        original_run_epoch = train._run_epoch
        original_gather = train._gather
        def run_epoch(model, loader, criterion, optimizer, scaler, device, task, training, *args, **kwargs):
            if not training:
                assert rank == 0
                assert not isinstance(model, torch.nn.parallel.DistributedDataParallel)
                assert len(loader.dataset) == 3
                assert not isinstance(loader.sampler, train.DistributedEvaluationSampler)
            return original_run_epoch(model, loader, criterion, optimizer, scaler, device,
                                      task, training, *args, **kwargs)
        def gather(value, distributed, world_size):
            # All-rank metric collection happens before validation and only
            # contains the local training result.
            assert len(value) == 5
            assert value[1] == 2
            return original_gather(value, distributed, world_size)
        train._run_epoch = run_epoch
        train._gather = gather
        train.main(cfg)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('backend', ['gloo', 'nccl'])
def test_two_rank_audio_workers_training_validation_and_gather(tmp_path, backend):
    if backend == 'nccl' and (not dist.is_nccl_available() or torch.cuda.device_count() < 2):
        pytest.skip('requires two CUDA devices and NCCL')
    manifest = tmp_path / 'audio.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['audio_path', 'label', 'label_split'])
        for split, counts in [(0, [10, 3, 9, 2]), (1, [10, 3, 9])]:
            for instance, count in enumerate(counts):
                for sentence in range(count):
                    name = f'split{split}-instance{instance}-s{sentence:04d}.wav'
                    sf.write(tmp_path / name, np.sin(np.arange(200) * .1 + sentence), 100)
                    writer.writerow([name, instance % 2 + 1, split])
    folder = tmp_path / 'output'
    mp.spawn(_train_worker, args=(backend, str(tmp_path / 'init'), str(folder), str(manifest)),
             nprocs=2, join=True)
    checkpoint = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 2
    assert len(checkpoint['history']) == 2
    with (folder / 'logs/eval_best_predict.csv').open() as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 3  # Rank 0 evaluates every validation instance once.
    for rank in range(2):
        log = (folder / ('train.log' if rank == 0 else 'train-rank1.log')).read_text()
        assert 'context=spawn' in log
        assert f'rank={rank} epoch=2 gathered training metrics' in log
        assert f'rank={rank} epoch=2 validation complete' in log
        assert (f'rank={rank} val epoch=2 finished samples=3' in log) == (rank == 0)
        if rank != 0:
            assert ' val epoch=' not in log
