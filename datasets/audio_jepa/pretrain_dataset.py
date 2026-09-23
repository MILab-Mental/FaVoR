"""Waveform manifests and crop collation for audio JEPA pre-training."""

from __future__ import annotations

import random
import os
import subprocess
from array import array
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def load_audio(path, sample_rate):
    try:
        waveform, source_rate = torchaudio.load(path)
    except (ImportError, RuntimeError) as torchaudio_error:
        try:
            array, source_rate = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(array.T)
        except Exception:
            # Some libsndfile builds cannot decode MP3. ffmpeg is the final
            # fallback and emits already-mono audio at the requested rate.
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                 "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
                check=False, capture_output=True,
            )
            if result.returncode or not result.stdout:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Could not decode {path}: {detail}") from torchaudio_error
            waveform = torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy()).unsqueeze(0)
            source_rate = sample_rate
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.squeeze(0)


def repeat_to_length(waveform, length):
    if waveform.numel() == 0:
        return torch.zeros(length)
    if waveform.numel() >= length:
        return waveform
    return waveform.repeat(length // waveform.numel() + 1)[:length]


def crop_waveform(waveform, length, *, start=None):
    waveform = repeat_to_length(waveform, length)
    if waveform.numel() == length:
        return waveform
    maximum = waveform.numel() - length
    start = random.randint(0, maximum) if start is None else max(0, min(int(start), maximum))
    return waveform[start:start + length]


def normalize_waveform(waveform, eps=1e-5):
    return (waveform - waveform.mean()) / (waveform.std() + eps)


def _manifest_path(line):
    """Read either a plain path or FAVOR's ``<path> <numeric-label>`` row."""
    line = line.strip()
    if not line:
        return None
    path, separator, suffix = line.rpartition(" ")
    if separator:
        try:
            float(suffix)
        except ValueError:
            pass
        else:
            return path
    return line


class AudioPretrainDataset(Dataset):
    def __init__(self, manifests, sample_rate=16000, process_seconds=2.0,
                 min_process_seconds=None, max_process_seconds=None,
                 samples_per_audio=1, normalize=True, max_retries=10):
        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        self.manifests = [Path(path).expanduser().resolve() for path in manifests]
        self.indices = [np.load(ensure_manifest_index(path), mmap_mode="r") for path in self.manifests]
        self.cumulative_sizes = []
        total = 0
        for index in self.indices:
            total += len(index)
            self.cumulative_sizes.append(total)
        self._handles = {}
        if not total:
            raise ValueError("No audio paths were found in the pre-training manifests")
        self.sample_rate = int(sample_rate)
        self.min_length = int(self.sample_rate * (min_process_seconds or process_seconds))
        self.max_length = int(self.sample_rate * (max_process_seconds or process_seconds))
        self.samples_per_audio = int(samples_per_audio)
        self.normalize = bool(normalize)
        self.max_retries = int(max_retries)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _audio_path(self, index):
        manifest_index = bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[manifest_index - 1] if manifest_index else 0
        local_index = index - previous
        path = self.manifests[manifest_index]
        # Each DataLoader worker opens its own descriptor on first access.
        handle = self._handles.get(manifest_index)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._handles[manifest_index] = handle
        handle.seek(int(self.indices[manifest_index][local_index]))
        value = _manifest_path(handle.readline().decode("utf-8"))
        if value is None:
            raise ValueError(f"Indexed an empty manifest row in {path}")
        return value

    def __getitem__(self, index):
        error = None
        for _ in range(self.max_retries):
            path = self._audio_path(index)
            try:
                waveform = load_audio(path, self.sample_rate)
                length = random.randint(self.min_length, self.max_length)
                crops = [crop_waveform(waveform, length) for _ in range(self.samples_per_audio)]
                if self.normalize:
                    crops = [normalize_waveform(crop) for crop in crops]
                return torch.stack(crops), length
            except Exception as exc:
                error = exc
                index = random.randrange(len(self))
        raise RuntimeError(f"Could not load a valid audio sample after {self.max_retries} attempts: {error}")


def _manifest_index_path(manifest):
    return Path(f"{manifest}.idx.npy")


def ensure_manifest_index(manifest):
    """Build a compact mmap-able array of line offsets for a large manifest."""
    manifest = Path(manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Audio manifest does not exist: {manifest}")
    index_path = _manifest_index_path(manifest)
    if index_path.is_file() and index_path.stat().st_mtime >= manifest.stat().st_mtime:
        return index_path
    offsets = array("Q")
    offset = 0
    with manifest.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    temporary = Path(f"{index_path}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(offsets, dtype=np.uint64), allow_pickle=False)
    os.replace(temporary, index_path)
    return index_path


def collate_audio_crops(batch):
    max_length = max(crops.shape[1] for crops, _ in batch)
    waveforms, lengths = [], []
    for crops, length in batch:
        if crops.shape[1] < max_length:
            crops = torch.nn.functional.pad(crops, (0, max_length - crops.shape[1]))
        waveforms.append(crops)
        lengths.extend([length] * crops.shape[0])
    return torch.cat(waveforms), torch.tensor(lengths, dtype=torch.long)


def make_audio_pretrain_loader(cfg, rank=0, world_size=1):
    manifests = cfg["datasets"]
    if isinstance(manifests, (str, Path)):
        manifests = [manifests]
    if rank == 0:
        for manifest in manifests:
            ensure_manifest_index(manifest)
    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        for manifest in manifests:
            ensure_manifest_index(manifest)
    dataset = AudioPretrainDataset(
        manifests,
        sample_rate=cfg.get("sample_rate", 16000),
        process_seconds=cfg.get("process_seconds", 2.0),
        min_process_seconds=cfg.get("min_process_seconds"),
        max_process_seconds=cfg.get("max_process_seconds"),
        samples_per_audio=cfg.get("samples_per_audio", 1),
        normalize=cfg.get("normalize_waveform", True),
        max_retries=cfg.get("max_retries", 10),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_audio_crops,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_mem", False),
        persistent_workers=bool(cfg.get("persistent_workers", False) and cfg.get("num_workers", 0) > 0),
    )
    return dataset, loader, sampler
