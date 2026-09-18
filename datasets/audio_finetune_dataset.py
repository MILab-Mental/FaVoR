"""CSV-supervised audio datasets following FAVOR's split and label contract."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.audio_pretrain_dataset import crop_waveform, load_audio, normalize_waveform


class AudioCSVDataset(Dataset):
    def __init__(self, csv_paths, split, *, root_paths, label_column, task,
                 num_class=None, sample_rate=16000, process_seconds=4.0,
                 num_clips=1, training=False, normalize=True, max_retries=10):
        if len(csv_paths) != len(root_paths):
            raise ValueError("data.datasets and data.rootpaths must have the same length")
        self.split = int(split)
        self.task = task.lower()
        self.label_column = label_column
        self.num_class = num_class
        self.sample_rate = int(sample_rate)
        self.crop_length = int(self.sample_rate * float(process_seconds))
        self.num_clips = int(num_clips)
        self.training = bool(training)
        self.normalize = bool(normalize)
        self.max_retries = int(max_retries)
        if self.task not in {"classification", "regression", "multi_label_classification"}:
            raise ValueError(f"Unsupported task: {task}")
        if self.task == "multi_label_classification" and (not isinstance(num_class, int) or num_class < 2):
            raise ValueError("multi_label_classification requires data.num_class >= 2")

        self.samples, self.labels = [], []
        split_column = f"{label_column}_split"
        for csv_value, root_value in zip(csv_paths, root_paths):
            csv_path = Path(csv_value).expanduser().resolve()
            root = Path(root_value).expanduser().resolve()
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"audio_path", label_column, split_column}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"Expected CSV columns {sorted(required)} in {csv_path}")
                for line_number, row in enumerate(reader, 2):
                    raw_split = str(row.get(split_column, "") or "").strip()
                    if raw_split in {"", "-1"}:
                        continue
                    if int(raw_split) != self.split:
                        continue
                    raw_path = str(row.get("audio_path", "") or "").strip()
                    raw_label = str(row.get(label_column, "") or "").strip()
                    if not raw_path or not raw_label:
                        raise ValueError(f"Empty audio path or label at {csv_path}:{line_number}")
                    if "|" in raw_path:
                        raise ValueError(f"Multi-path audio rows are not yet supported: {csv_path}:{line_number}")
                    path = Path(raw_path).expanduser()
                    if not path.is_absolute():
                        path = root / path
                    self.samples.append(str(path.resolve()))
                    self.labels.append(self._parse_label(raw_label, csv_path, line_number))
        if not self.samples:
            raise ValueError(f"No samples with split={split} found in {csv_paths}")

    def _parse_label(self, value, csv_path, line_number):
        if self.task == "classification":
            return int(value) - 1
        if self.task == "regression":
            return float(value)
        target = np.zeros(self.num_class, dtype=np.float32)
        for token in value.split("|"):
            index = int(token.strip()) - 1
            if index < 0 or index >= self.num_class:
                raise ValueError(f"Class {token!r} out of range at {csv_path}:{line_number}")
            target[index] = 1.0
        return target

    def __len__(self):
        return len(self.samples)

    def _clips(self, waveform):
        maximum = max(0, waveform.numel() - self.crop_length)
        clips = []
        for clip_index in range(self.num_clips):
            start = None if self.training else int(maximum * clip_index / max(1, self.num_clips - 1))
            clip = crop_waveform(waveform, self.crop_length, start=start)
            clips.append(normalize_waveform(clip) if self.normalize else clip)
        return torch.stack(clips)

    def __getitem__(self, index):
        error = None
        for _ in range(self.max_retries):
            path = self.samples[index]
            try:
                clips = self._clips(load_audio(path, self.sample_rate))
                label = self.labels[index]
                if self.task == "classification":
                    label = torch.tensor(label, dtype=torch.long)
                else:
                    label = torch.as_tensor(label, dtype=torch.float32)
                return clips, label, path
            except Exception as exc:
                error = exc
                index = random.randrange(len(self.samples))
        raise RuntimeError(f"Could not load a valid audio sample after {self.max_retries} attempts: {error}")


def make_audiodataset_finetune_a(csv_paths, *, root_paths, label_column, task,
                                 num_class, sample_rate, process_seconds, num_clips):
    common = dict(
        root_paths=root_paths, label_column=label_column, task=task, num_class=num_class,
        sample_rate=sample_rate, process_seconds=process_seconds, num_clips=num_clips,
    )
    return (
        AudioCSVDataset(csv_paths, 0, training=True, **common),
        AudioCSVDataset(csv_paths, 1, training=False, **common),
    )

