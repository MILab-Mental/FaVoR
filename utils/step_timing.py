"""Opt-in synchronized wall-clock timing for short diagnostic training runs."""
import csv
import time
from collections import Counter
from contextlib import contextmanager, nullcontext

import torch


class StepTiming:
    # Forward children overlap with forward_seconds; do not sum both.
    phases = ('data_wait', 'batch_sync', 'to_device', 'forward', 'video_encoder',
              'audio_encoder', 'alignment', 'fusion', 'projector', 'loss',
              'loss_check', 'backward', 'grad_stats_clip', 'optimizer', 'ema',
              'logging', 'checkpoint')

    def __init__(self, device, enabled=False):
        self.device = device
        self.enabled = enabled
        self.reset()

    def synchronize(self):
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def reset(self):
        self.seconds = Counter()
        self.counts = Counter()
        self.started = time.perf_counter()

    def phase(self, name):
        return self._phase(name) if self.enabled else nullcontext()

    @contextmanager
    def _phase(self, name):
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] += time.perf_counter() - started
            self.counts[name] += 1

    def batches(self, loader):
        # Include worker startup in the first data_wait, and epoch transitions
        # in the current optimizer step when accumulating across epochs.
        with self.phase('data_wait'):
            iterator = iter(loader)
        while True:
            with self.phase('data_wait'):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch

    def report(self, path, rank, step, epoch, micro_batches, samples):
        self.synchronize()
        total = time.perf_counter() - self.started
        row = dict(step=step, epoch=epoch, rank=rank, micro_batches=micro_batches,
                   samples=samples, total_seconds=total,
                   **{name + '_seconds': self.seconds[name] for name in self.phases},
                   encoder_calls=self.counts['video_encoder'])
        exists = path.exists() and path.stat().st_size > 0
        with path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        values = ' '.join(f'{name}={self.seconds[name]:.3f}s' for name in self.phases)
        # The main launcher suppresses rank>0 loggers. Print both ranks so slow
        # rank/data imbalance is visible without extra diagnostic collectives.
        print(f'[timing rank={rank} step={step}] total={total:.3f}s '
              f'micros={micro_batches} samples={samples} '
              f'encoder_calls={self.counts["video_encoder"]} {values}', flush=True)
