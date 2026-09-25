"""Anchor and local-view construction independent of audio file decoding."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from datasets.audio_jepa.pretrain_dataset import repeat_to_length
from .augment import augment_local_waveform


def make_audio_views(
    waveform,
    *,
    sample_rate=16000,
    global_seconds=4.0,
    local_seconds=2.0,
    local_views=4,
    short_audio_policy="repeat",
    normalize=True,
    augmentation=None,
    view_mode="crop_augment",
):
    global_length = int(round(float(global_seconds) * sample_rate))
    local_length = int(round(float(local_seconds) * sample_rate))
    if global_length <= 0 or local_length <= 0 or local_length > global_length:
        raise ValueError("audio view durations must satisfy 0 < local <= global")
    source_length = waveform.numel()
    was_repeated = was_padded = False
    if source_length < global_length:
        if short_audio_policy == "skip":
            raise ValueError("audio is shorter than the global window")
        if short_audio_policy == "repeat":
            anchor = repeat_to_length(waveform, global_length)
            anchor_valid_length = global_length
            was_repeated = True
        elif short_audio_policy == "zero_pad":
            anchor = F.pad(waveform, (0, global_length - source_length))
            anchor_valid_length = source_length
            was_padded = True
        else:
            raise ValueError("short_audio_policy must be repeat, zero_pad, or skip")
        anchor_start = 0
    else:
        anchor_start = random.randint(0, source_length - global_length)
        anchor = waveform[anchor_start:anchor_start + global_length].clone()
        anchor_valid_length = global_length
    if normalize:
        valid = anchor[:anchor_valid_length]
        mean = valid.mean() if valid.numel() else anchor.new_tensor(0)
        std = valid.std() if valid.numel() > 1 else anchor.new_tensor(1)
        anchor = (anchor - mean) / (std + 1e-5)
        if was_padded:
            anchor[anchor_valid_length:] = 0

    if view_mode == "augmentation_only":
        local_length = global_length
    elif view_mode not in {"crop_only", "crop_augment"}:
        raise ValueError("view_mode must be augmentation_only, crop_only, or crop_augment")
    local_audio, local_lengths, local_intervals, occurrence = [], [], [], []
    for _ in range(int(local_views)):
        start = random.randint(0, global_length - local_length)
        local = anchor[start:start + local_length].clone()
        valid_length = max(0, min(local_length, anchor_valid_length - start))
        stats = {"gain": False, "speed": False, "polarity": False, "time_mask": False}
        if view_mode in {"augmentation_only", "crop_augment"}:
            local, stats = augment_local_waveform(local, augmentation or {})
        local_audio.append(local)
        local_lengths.append(valid_length if was_padded else local_length)
        local_intervals.append((start, start + local_length))
        occurrence.append([float(stats[name]) for name in ("gain", "speed", "polarity", "time_mask")])
    return {
        "global_audio": anchor,
        "global_length": anchor_valid_length,
        "local_audio": torch.stack(local_audio),
        "local_lengths": torch.tensor(local_lengths, dtype=torch.long),
        "local_intervals": torch.tensor(local_intervals, dtype=torch.long),
        # This interval is in the original file.  Repeated/padded samples keep
        # their true source endpoint; local_intervals remain relative to the
        # constructed global view.
        "anchor_interval": torch.tensor(
            (anchor_start, anchor_start + min(source_length, global_length)), dtype=torch.long
        ),
        "was_repeated": was_repeated,
        "was_padded": was_padded,
        "augmentation_occurrence": torch.tensor(occurrence, dtype=torch.float32),
    }

