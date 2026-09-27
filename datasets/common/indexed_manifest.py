"""Reusable, validated JSONL records with mmap offsets and worker-local readers."""
import fcntl
import hashlib
import json
import logging
import os
import operator
import shutil
import tempfile
from array import array
from collections.abc import Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def _signature(paths, kind, options):
    return dict(version=2, kind=kind, options=options or {}, sources=[
        dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
             ctime_ns=path.stat().st_ctime_ns) for path in paths])


class IndexedManifest(Sequence):
    """Parse/validate once, then retrieve one record at a time in original order.

    The reader callable is used only during cache construction, never stored or
    sent to DataLoader workers. Cache generations are immutable and atomically
    published so existing workers keep reading their original dataset snapshot.
    """

    def __init__(self, paths, reader, *, kind, options=None, unique_key=None):
        paths = [Path(path).expanduser().resolve() for path in paths]
        if not paths:
            raise ValueError('no manifest paths')
        signature = _signature(paths, kind, options)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        cache_root = Path(str(paths[0]) + '.index')
        self.directory = cache_root / key
        cache_root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.directory / 'metadata.json'
        if not metadata_path.is_file():
            # Works for independent launches too, without a distributed barrier
            # that could strand another rank if validation fails on the writer.
            with (cache_root / 'build.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not metadata_path.is_file():
                    LOGGER.info('Building %s manifest index: %s', kind, paths)
                    temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=cache_root))
                    try:
                        offsets, counts, seen = array('Q'), [], set()
                        with (temporary / 'records.jsonl').open('wb') as handle:
                            for path in paths:
                                count = 0
                                for row in reader(path):
                                    if unique_key is not None:
                                        identity = row[unique_key]
                                        if identity in seen:
                                            raise ValueError(f'duplicate {unique_key} across manifests: {identity}')
                                        seen.add(identity)
                                    offsets.append(handle.tell())
                                    handle.write((json.dumps(list(row.items()), ensure_ascii=False, default=_json_scalar) + '\n').encode('utf-8'))
                                    count += 1
                                    if len(offsets) % 100000 == 0:
                                        LOGGER.info('Indexed %s: %d records', kind, len(offsets))
                                counts.append(count)
                        if not offsets:
                            raise ValueError('empty manifests')
                        if _signature(paths, kind, options) != signature:
                            raise RuntimeError('manifest changed while building index; retry with a stable source')
                        np.save(temporary / 'offsets.npy', np.asarray(offsets, dtype=np.uint64), allow_pickle=False)
                        (temporary / 'metadata.json').write_text(json.dumps(dict(signature=signature, counts=counts)), encoding='utf-8')
                        os.replace(temporary, self.directory)
                        LOGGER.info('Built %s manifest index: %d records at %s', kind, len(offsets), self.directory)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        self.counts = metadata['counts']
        self.size = sum(self.counts)
        self._offsets = self._handle = self._pid = None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._offsets = np.load(self.directory / 'offsets.npy', mmap_mode='r', allow_pickle=False)
            self._handle = (self.directory / 'records.jsonl').open('rb')
            self._pid = os.getpid()
        self._handle.seek(int(self._offsets[index]))
        return dict(json.loads(self._handle.readline()))

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._offsets = self._handle = self._pid = None

    def __getstate__(self):
        # np.memmap's default pickle includes the array contents. Reopen the
        # mmap and file separately in each worker instead (fork or spawn).
        return dict(self.__dict__, _offsets=None, _handle=None, _pid=None)

    def __del__(self):
        if hasattr(self, '_handle'):
            self.close()


class ManifestColumn(Sequence):
    def __init__(self, records, key):
        self.records, self.key = records, key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [row[self.key] for row in self.records[index]]
        return self.records[index][self.key]


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported manifest value: {type(value).__name__}')
