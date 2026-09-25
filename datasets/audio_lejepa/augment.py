"""Mild waveform-only local augmentations for AUDIO-LeJEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _draw(generator=None):
    return float(torch.rand((), generator=generator))


def _enabled(config, generator):
    return bool(config.get("enabled", False)) and _draw(generator) < float(config.get("probability", 1.0))


def augment_local_waveform(waveform, cfg, *, generator=None):
    """Return a fixed-length augmented waveform and occurrence indicators."""
    output = waveform.clone()
    target_length = output.numel()
    occurred = {"gain": False, "speed": False, "polarity": False, "time_mask": False}

    gain = cfg.get("gain", {})
    if _enabled(gain, generator):
        minimum = float(gain.get("min_db", -6.0))
        maximum = float(gain.get("max_db", 6.0))
        db = minimum + (maximum - minimum) * _draw(generator)
        output.mul_(10.0 ** (db / 20.0))
        occurred["gain"] = True

    speed = cfg.get("speed", {})
    if _enabled(speed, generator):
        minimum = float(speed.get("min_rate", 0.95))
        maximum = float(speed.get("max_rate", 1.05))
        rate = minimum + (maximum - minimum) * _draw(generator)
        changed_length = max(1, int(round(target_length / rate)))
        output = F.interpolate(
            output.view(1, 1, -1), size=changed_length, mode="linear", align_corners=False
        ).view(-1)
        if changed_length < target_length:
            output = F.pad(output, (0, target_length - changed_length))
        elif changed_length > target_length:
            maximum_start = changed_length - target_length
            start = int(_draw(generator) * (maximum_start + 1))
            output = output[start:start + target_length]
        occurred["speed"] = True

    polarity = cfg.get("polarity", {})
    if _enabled(polarity, generator):
        output.neg_()
        occurred["polarity"] = True

    time_mask = cfg.get("time_mask", {})
    if _enabled(time_mask, generator):
        max_ratio = float(time_mask.get("max_ratio", 0.10))
        if not 0 <= max_ratio <= 0.10:
            raise ValueError("audio_augmentation.time_mask.max_ratio must be in [0,0.10]")
        width = max(1, int(round(target_length * max_ratio * _draw(generator))))
        start = int(_draw(generator) * max(target_length - width + 1, 1))
        output[start:start + width] = 0
        occurred["time_mask"] = True
    return output, occurred

