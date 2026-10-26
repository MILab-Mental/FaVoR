import pytest

from utils.early_stopping import early_stopping_patience, stale_epochs_from_history


def test_default_and_disabled_patience():
    assert early_stopping_patience({}) == 10
    assert early_stopping_patience({'optimization': {'early_stopping_patience': 0}}) == 0
    assert early_stopping_patience({'optimization': {'early_stopping_patience': 10},
                                    'finetune': {'early_stopping_patience': 2}}) == 10
    assert early_stopping_patience({'finetune': {'early_stopping_patience': 0}}) == 0
    with pytest.raises(ValueError):
        early_stopping_patience({'optimization': {'early_stopping_patience': -1}})


@pytest.mark.parametrize('maximize,values,expected', [
    (True, [.4, .6, .6, .5], 2),
    (True, [.4, .6, .5, .7], 0),
    (False, [3., 2., 2., 2.5], 2),
    (True, [.6, float('nan'), .5], 2),
    (True, [], 0),
])
def test_legacy_resume_counts_only_evaluated_scores(maximize, values, expected):
    history = [{'metric': value} for value in values]
    history.insert(1, {'epoch': 2})
    assert stale_epochs_from_history(history, 'metric', maximize) == expected


def _video_worker(rank, init_file, folder):
    import torch
    import torch.distributed as dist
    import app.finetune_v.train as train
    from datetime import timedelta
    from pathlib import Path
    from test_sentence_instances import instance_pair, tiny_video, InstancePredictionHead

    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        aggregator = dict(pooling='attention', hidden_dim=8)
        train.make_videodataset_finetune_v = lambda *a, **k: instance_pair('video')
        train.build_model = lambda *a, **k: (
            tiny_video(), InstancePredictionHead(24, 2, aggregator))
        original_metrics = train.classification_metrics
        def constant_score(*args, **kwargs):
            result = original_metrics(*args, **kwargs)
            result[0]['f1_macro'] = .5
            return result
        train.classification_metrics = constant_score
        config = dict(
            folder=folder, meta=dict(dtype='float32', eval_freq=2),
            data=dict(task='classification', num_class=2, batch_size=1, num_workers=0,
                      datasets=['unused'], label_column='emotion', num_sentence=8,
                      crop_size=16, patch_size=8, tubelet_size=1, dataset_fpcs=[4]),
            optimization=dict(early_stopping_patience=1, epochs=8, lr=.001, warmup=0, start_lr=.001, final_lr=0,
                              weight_decay=0, final_weight_decay=0),
            model=dict(model_name='tiny', out_layers=[0, 1], instance_aggregator=aggregator))
        train.main(config)
        if rank == 0:
            path = Path(folder) / 'latest.pt'
            state = torch.load(path, weights_only=False)
            assert state['epoch'] == 4
            assert state['stale_epochs'] == 1
            assert len(state['history']) == 2
            # Old checkpoints lack the counter; resume must infer it and stop.
            del state['stale_epochs']
            torch.save(state, path)
        dist.barrier()
        train.main(config)
        assert torch.load(Path(folder) / 'latest.pt', weights_only=False)['epoch'] == 4
    finally:
        dist.destroy_process_group()


def test_video_distributed_early_stop_skipped_evaluation_and_legacy_resume(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_video_worker, args=(str(tmp_path / 'init'), str(tmp_path / 'video')),
             nprocs=2, join=True)
