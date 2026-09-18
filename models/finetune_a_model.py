"""Audio downstream backbone and task heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.audio_jepa import AudioBackbone

class MultiClipAttentionHead(nn.Module):
    def __init__(self, embed_dim, num_heads, task, num_class=None, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        output_dim = num_class if task in {"classification", "multi_label_classification"} else 1
        self.output = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, clip_features):
        query = self.query.expand(clip_features.shape[0], -1, -1)
        pooled, _ = self.attention(query, clip_features, clip_features, need_weights=False)
        return self.output(self.norm(pooled.squeeze(1)))


class AudioFineTuneModel(nn.Module):
    def __init__(self, cfg, task, num_class=None):
        super().__init__()
        self.backbone = AudioBackbone(cfg)
        head_cfg = cfg.get("head", {})
        self.head = MultiClipAttentionHead(
            self.backbone.embed_dim,
            int(head_cfg.get("num_heads", 4)),
            task,
            num_class=num_class,
            hidden_dim=int(head_cfg.get("hidden_dim", 256)),
            dropout=float(head_cfg.get("dropout", 0.1)),
        )

    def forward(self, clips):
        batch_size, num_clips, length = clips.shape
        features = self.backbone.extract_features(clips.reshape(batch_size * num_clips, length), pool=True)
        return self.head(features.reshape(batch_size, num_clips, -1))
