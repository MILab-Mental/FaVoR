"""Audio JEPA model factory."""

from __future__ import annotations

from models.audio_jepa import AudioJEPA


def build_audio_jepa(cfg):
    return AudioJEPA(cfg)
