import csv
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from datasets.common.indexed_manifest import IndexedManifest
from datasets.common.weighted_sampler import DistributedWeightedSampler
from datasets.va_lejepa import VALeJEPADataset
from datasets.va_lejepa.va_manifest import read_manifest
from datasets.video_lejepa.dataset import VideoLeJEPADataset


def _write_va(path, names, **extra):
    rows = [dict(pair_id=name, source_id=name, video_path=f'video/{name}.mp4',
                 audio_path=f'audio/{name}.wav', **extra) for name in names]
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_va_index_reuses_validation_and_preserves_csv_rows(tmp_path, monkeypatch):
    path = tmp_path/'pairs.csv'
    _write_va(path, ['a', 'b', 'c'], label='first\n第二行')
    expected = read_manifest(path)
    dataset = VALeJEPADataset(dict(manifests=[str(path)]))
    assert isinstance(dataset.rows, IndexedManifest)
    assert list(dataset.rows) == expected
    assert dataset.rows[-1] == expected[-1] and dataset.rows[1:] == expected[1:]
    with pytest.raises(IndexError): dataset.rows[3]
    with pytest.raises(IndexError): dataset.rows[-4]
    import datasets.va_lejepa.dataset as module
    monkeypatch.setattr(module, 'iter_manifest', lambda *a, **k: (_ for _ in ()).throw(AssertionError('reparsed cache')))
    second = VALeJEPADataset(dict(manifests=[str(path)]))
    assert second.rows.directory == dataset.rows.directory
    assert list(second.rows) == expected


def test_va_index_invalidates_changed_source_and_options(tmp_path):
    path = tmp_path/'pairs.csv'
    _write_va(path, ['a'])
    first = VALeJEPADataset(dict(manifests=[str(path)]))
    rooted = VALeJEPADataset(dict(manifests=[str(path)], root=str(tmp_path/'other')))
    assert rooted.rows.directory != first.rows.directory
    assert rooted.rows[0]['video_path'] == str(tmp_path/'other/video/a.mp4')
    _write_va(path, ['a', 'b'])
    second = VALeJEPADataset(dict(manifests=[str(path)]))
    assert len(second) == 2 and second.rows.directory != first.rows.directory
    # Existing workers keep the original immutable generation.
    assert len(first) == 1 and first.rows[0]['pair_id'] == 'a'
    with pytest.raises(ValueError, match='extract source identity'):
        VALeJEPADataset(dict(manifests=[str(path)], source_pattern='does-not-match'))


def test_va_duplicate_across_manifests_and_failed_build_not_published(tmp_path):
    first, second = tmp_path/'one.csv', tmp_path/'two.csv'
    _write_va(first, ['a']); _write_va(second, ['a'])
    with pytest.raises(ValueError, match='duplicate pair_id across manifests'):
        VALeJEPADataset(dict(manifests=[str(first), str(second)]))
    assert not list(Path(str(first)+'.index').glob('*/metadata.json'))
    _write_va(second, ['b'])
    dataset = VALeJEPADataset(dict(manifests=[str(first), str(second)]))
    assert [r['pair_id'] for r in dataset.rows] == ['a', 'b']
    assert dataset.rows.counts == [1, 1]


def test_va_path_resolution_keeps_parent_and_file_symlink_semantics(tmp_path):
    real = tmp_path/'real'; real.mkdir()
    (real/'a.mp4').touch(); (real/'a.wav').touch()
    (tmp_path/'linked').symlink_to(real, target_is_directory=True)
    (tmp_path/'a.mp4').symlink_to(real/'a.mp4')
    path = tmp_path/'pairs.csv'
    path.write_text('pair_id,source_id,video_path,audio_path\na,a,a.mp4,linked/a.wav\n')
    row = read_manifest(path)[0]
    assert row['video_path'] == str((tmp_path/'a.mp4').resolve())
    assert row['audio_path'] == str((tmp_path/'linked/a.wav').resolve())
    assert VALeJEPADataset(dict(manifests=[str(path)])).rows[0] == row


def test_video_index_preserves_labels_dataset_mapping_and_weighted_sampling(tmp_path):
    first, second = tmp_path/'videos.csv', tmp_path/'images.npy'
    first.write_text('/data/a.mp4 0\n"/data/b space.mp4" 1\n')
    np.save(second, np.array(['/data/c.jpg'], dtype=object))
    dataset = VideoLeJEPADataset(data_paths=[str(first), str(second)], datasets_weights=[2., 3.],
                               dataset_fpcs=[4, 4], fps=8, frame_step=None)
    assert isinstance(dataset.samples.records, IndexedManifest)
    assert list(dataset.samples) == ['/data/a.mp4', '/data/b space.mp4', '/data/c.jpg']
    assert list(dataset.labels) == [0, 1, 0]
    assert dataset.num_samples_per_dataset == [2, 1]
    assert [dataset.per_dataset_indices[i] for i in range(3)] == [(0, 0), (0, 1), (1, 0)]
    class OldWeights:
        sample_weights = [1., 1., 3.]
        def __len__(self): return 3
    for rank in (0, 1):
        reference = DistributedWeightedSampler(OldWeights(), num_replicas=2, rank=rank, seed=17)
        actual = DistributedWeightedSampler(dataset, num_replicas=2, rank=rank, seed=17)
        for epoch in (0, 4):
            reference.set_epoch(epoch); actual.set_epoch(epoch)
            assert list(actual) == list(reference)


@pytest.mark.parametrize('context', ['fork', 'spawn'])
def test_index_worker_local_handles_and_small_pickle(tmp_path, context):
    path = tmp_path/'source.txt'; path.write_text('source')
    records = IndexedManifest([path], lambda p: (dict(index=i, text='行') for i in range(1000)), kind='test-workers')
    assert records[7]['index'] == 7  # Open handles before forking/pickling.
    assert isinstance(records._offsets, np.memmap)
    payload = pickle.dumps(records)
    assert len(payload) < 2048
    restored = pickle.loads(payload)
    assert restored._handle is None and restored[31]['index'] == 31
    loader = DataLoader(records, batch_size=25, num_workers=2, multiprocessing_context=context)
    assert [int(i) for batch in loader for i in batch['index']] == list(range(1000))
    assert records[9]['index'] == 9


def _concurrent_builder(path, marker, queue):
    def read(p):
        with open(marker, 'a') as handle: handle.write('built\n')
        yield dict(value=3)
    records = IndexedManifest([path], read, kind='concurrent')
    queue.put(records[0])


def test_concurrent_build_publishes_one_generation(tmp_path):
    path = tmp_path/'source.txt'; path.write_text('source')
    marker = tmp_path/'marker.txt'
    context = mp.get_context('spawn')
    queue = context.Queue()
    workers = [context.Process(target=_concurrent_builder, args=(str(path), str(marker), queue)) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate(); worker.join()
            pytest.fail('concurrent index build stalled')
        assert worker.exitcode == 0
    assert [queue.get(timeout=5) for _ in workers] == [dict(value=3), dict(value=3)]
    assert marker.read_text().splitlines() == ['built']
