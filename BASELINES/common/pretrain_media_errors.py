"""Shared, per-modality records of unreadable pretraining media."""
import csv
import fcntl
import time
from contextlib import contextmanager

from BASELINES.common.runtime import OUTPUT_ROOT


MODALITY_SUFFIX = {'audio': 'a', 'video': 'v', 'audio-video': 'va'}
FIELDS = ['name', 'reason']


@contextmanager
def _writer_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.error_media.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _read(filename):
    try:
        stream = filename.open(encoding='utf-8-sig', newline='')
    except FileNotFoundError:
        return {}
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Unexpected columns in {filename}, expected name,reason')
        return {row['name']: row['reason'] for row in reader if row['name']}


def _write(filename, rows):
    temporary = filename.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows.items())
    temporary.replace(filename)


class PretrainMediaErrors:
    def __init__(self, modality, directory=None, *, writable=True):
        directory = directory if directory is not None else OUTPUT_ROOT / 'pretrain'
        self.filename = directory / f'error_media_{MODALITY_SUFFIX[modality]}.csv'
        self.writable = writable
        self.names = set()
        self._signature = None
        self._next_check = 0.
        if writable:
            tables = [directory / f'error_media_{suffix}.csv' for suffix in MODALITY_SUFFIX.values()]
            if not all(table.is_file() for table in tables):
                with _writer_lock(directory):
                    for table in tables:
                        if not table.is_file():
                            _write(table, {})

    def known(self, *, refresh=False):
        now = time.monotonic()
        if refresh or now >= self._next_check:
            self._next_check = now + 5
            try:
                stat = self.filename.stat()
                signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                signature = None
            if signature != self._signature:
                self.names = set(_read(self.filename))
                self._signature = signature
        return self.names

    def record(self, name, reason):
        if not name or name in self.names:
            return
        if self.writable:
            with _writer_lock(self.filename.parent):
                rows = _read(self.filename)
                if name not in rows:
                    rows[name] = reason
                    _write(self.filename, rows)
                self.names.update(rows)
        self.names.add(name)
