"""Shared-backbone AUDIO-LeJEPA model."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import AudioLeEncoder
from models.video_lejepa import Projector


class AudioLeJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AudioLeEncoder(cfg)
        projector_cfg = cfg.get("projector", {})
        self.projector = Projector(
            self.encoder.embed_dim,
            hidden_dim=int(projector_cfg.get("hidden_dim", 2048)),
            output_dim=int(projector_cfg.get("output_dim", 256)),
        )

    def forward(self, global_audio, global_lengths, local_audio, local_lengths):
        if local_audio.ndim != 3:
            raise ValueError(
                f"local_audio must be [B,K,L], got {tuple(local_audio.shape)}"
            )
        batch, views, samples = local_audio.shape
        global_embedding = self.encoder(global_audio, global_lengths).unsqueeze(1)
        local_embedding = self.encoder(
            local_audio.reshape(batch * views, samples),
            local_lengths.reshape(batch * views),
        ).reshape(batch, views, -1)
        return self.projector(torch.cat((global_embedding, local_embedding), dim=1))


def build_audio_lejepa(model_cfg):
    return AudioLeJEPA(model_cfg)

