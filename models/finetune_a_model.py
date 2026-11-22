"""Audio downstream backbone and task heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.audio_jepa import AudioBackbone
from models.instance_aggregator import InstancePredictionHead
from models.regression_head import LayerWeightedRegressor


class MultiClipAttentionHead(nn.Module):
    def __init__(self, embed_dim, num_heads, task, num_class=None, hidden_dim=256,
                 dropout=0.1, attention_dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True,
        )
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
        self.instance_enabled = 'instance_aggregator' in cfg
        regression_head = cfg.get("regression_head", "attentive")
        if regression_head not in {"attentive", "layer_weighted"}:
            raise ValueError("model.regression_head must be attentive or layer_weighted")
        self.layer_weighted = regression_head == "layer_weighted"
        if self.layer_weighted:
            if task != "regression" or self.instance_enabled:
                raise ValueError("model.regression_head=layer_weighted requires audio regression without instance aggregation")
            num_layers = len(self.backbone.context_encoder.layers)
            if num_layers < 1:
                raise ValueError("model.regression_head=layer_weighted requires at least one encoder block")
            self.head = LayerWeightedRegressor(
                self.backbone.embed_dim, num_layers,
                dropout=float(head_cfg.get("dropout", 0.1)),
                hidden_dim=int(head_cfg.get("hidden_dim", 256)),
            )
            return
        if self.instance_enabled:
            outputs = num_class if task in {'classification', 'multi_label_classification'} else 1
            self.head = InstancePredictionHead(self.backbone.embed_dim, outputs, cfg['instance_aggregator'])
            return
        self.head = MultiClipAttentionHead(
            self.backbone.embed_dim,
            int(head_cfg.get("num_heads", 4)),
            task,
            num_class=num_class,
            hidden_dim=int(head_cfg.get("hidden_dim", 256)),
            dropout=float(head_cfg.get("dropout", 0.1)),
            attention_dropout=float(head_cfg.get("attention_dropout", 0.0)),
        )

    def forward(self, clips):
        if self.instance_enabled:
            sentences = clips if clips.ndim == 4 else clips.unsqueeze(1)
            batch, count, crops, length = sentences.shape
            features = self.backbone.extract_features(sentences.reshape(batch * count * crops, length), pool=True)
            features = features.reshape(batch, count, crops, -1).mean(2)
            return self.head(features)
        batch_size, num_clips, length = clips.shape
        waveforms = clips.reshape(batch_size * num_clips, length)
        if self.layer_weighted:
            _, layers, _ = self.backbone(waveforms, return_all_layers=True, return_block_outputs=True)
            # Each layer contributes one feature per clip, then each clip contributes equally.
            pooled = torch.stack([tokens.mean(dim=1, dtype=torch.float32) for tokens in layers], dim=1)
            layer_features = pooled.reshape(batch_size, num_clips, len(layers), -1).mean(dim=1)
            return self.head(layer_features)
        features = self.backbone.extract_features(waveforms, pool=True)
        return self.head(features.reshape(batch_size, num_clips, -1))
