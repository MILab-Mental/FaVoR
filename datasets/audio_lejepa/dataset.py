"""Manifest-backed AUDIO-LeJEPA global/local dataset and variable-length collate."""

from __future__ import annotations

import random
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.audio_jepa.pretrain_dataset import (
    _manifest_path,
    ensure_manifest_index,
    load_audio,
)
from .views import make_audio_views


class AudioLeJEPADataset(Dataset):
    def __init__(self, cfg):
        manifests = cfg["datasets"]
        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        self.manifests = [Path(path).expanduser().resolve() for path in manifests]
        self.indices = [np.load(ensure_manifest_index(path), mmap_mode="r") for path in self.manifests]
        self.cumulative_sizes, total = [], 0
        for index in self.indices:
            total += len(index)
            self.cumulative_sizes.append(total)
        if not total:
            raise ValueError("No audio paths were found in the pre-training manifests")
        self._handles = {}
        self.cfg = cfg

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _audio_path(self, index):
        manifest_index = bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[manifest_index - 1] if manifest_index else 0
        handle = self._handles.get(manifest_index)
        if handle is None or handle.closed:
            handle = self.manifests[manifest_index].open("rb")
            self._handles[manifest_index] = handle
        handle.seek(int(self.indices[manifest_index][index - previous]))
        return _manifest_path(handle.readline().decode("utf-8"))

    def __getitem__(self, index):
        error = None
        for _ in range(int(self.cfg.get("max_retries", 10))):
            path = self._audio_path(index)
            try:
                waveform = load_audio(path, int(self.cfg.get("sample_rate", 16000)))
                sample = make_audio_views(
                    waveform,
                    sample_rate=int(self.cfg.get("sample_rate", 16000)),
                    global_seconds=float(self.cfg.get("global_seconds", 4.0)),
                    local_seconds=float(self.cfg.get("local_seconds", 2.0)),
                    local_views=int(self.cfg.get("local_views", 4)),
                    short_audio_policy=self.cfg.get("short_audio_policy", "repeat"),
                    normalize=bool(self.cfg.get("normalize_waveform", True)),
                    augmentation=self.cfg.get("audio_augmentation", {}),
                    view_mode=self.cfg.get("view_mode", "crop_augment"),
                )
                sample["path"] = path
                return sample
            except Exception as exc:
                error = exc
                index = random.randrange(len(self))
        raise RuntimeError(f"Could not load a valid audio sample: {error}")


def collate_audio_lejepa(samples):
    global_max = max(item["global_audio"].numel() for item in samples)
    local_max = max(item["local_audio"].shape[-1] for item in samples)
    globals_, locals_ = [], []
    for item in samples:
        globals_.append(torch.nn.functional.pad(
            item["global_audio"], (0, global_max - item["global_audio"].numel())
        ))
        locals_.append(torch.nn.functional.pad(
            item["local_audio"], (0, local_max - item["local_audio"].shape[-1])
        ))
    return {
        "global_audio": torch.stack(globals_),
        "global_lengths": torch.tensor([item["global_length"] for item in samples], dtype=torch.long),
        "local_audio": torch.stack(locals_),
        "local_lengths": torch.stack([item["local_lengths"] for item in samples]),
        "local_intervals": torch.stack([item["local_intervals"] for item in samples]),
        "anchor_intervals": torch.stack([item["anchor_interval"] for item in samples]),
        "was_repeated": torch.tensor([item["was_repeated"] for item in samples]),
        "was_padded": torch.tensor([item["was_padded"] for item in samples]),
        "augmentation_occurrence": torch.stack([item["augmentation_occurrence"] for item in samples]),
        "paths": [item["path"] for item in samples],
    }


def make_audio_lejepa_loader(cfg, rank=0, world_size=1):
    manifests = cfg["datasets"] if isinstance(cfg["datasets"], list) else [cfg["datasets"]]
    if rank == 0:
        for manifest in manifests:
            ensure_manifest_index(manifest)
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    dataset = AudioLeJEPADataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=sampler,
        drop_last=bool(cfg.get("drop_last", True)),
        collate_fn=collate_audio_lejepa,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_mem", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False) and cfg.get("num_workers", 0) > 0),
    )
    return dataset, loader, sampler

