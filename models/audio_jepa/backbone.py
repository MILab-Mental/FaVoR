"""Reusable waveform backbone shared by audio pre-training and fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from .extractor import ConvFeatureExtractor
from .positional import FixedPositionalEmbedding
from .target import topk_average
from .transformer import PreNormTransformerEncoder


class AudioBackbone(nn.Module):
    """emotion2vec-compatible CNN and Transformer without JEPA-only modules."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        audio_cfg = cfg["audio"]
        extractor_cfg = cfg["extractor"]
        encoder_cfg = cfg["encoder"]

        self.feature_extractor = ConvFeatureExtractor(
            conv_layers=extractor_cfg["conv_layers"],
            mode=extractor_cfg.get("mode", "layer_norm"),
            conv_bias=extractor_cfg.get("conv_bias", False),
            dropout=extractor_cfg.get("dropout", 0.0),
        )
        extractor_dim = self.feature_extractor.embedding_dim
        self.embed_dim = int(encoder_cfg["d_model"])
        self.feature_norm = nn.LayerNorm(extractor_dim)
        self.post_extraction_mapper = nn.Linear(extractor_dim, self.embed_dim)

        sample_rate = int(audio_cfg["sample_rate"])
        max_seconds = float(audio_cfg.get("max_process_seconds", audio_cfg["process_seconds"]))
        self.max_tokens = self.feature_extractor.output_length(int(sample_rate * max_seconds))
        self.pos_embed_encoder = FixedPositionalEmbedding(self.max_tokens, self.embed_dim)
        self.context_encoder = PreNormTransformerEncoder(
            depth=int(encoder_cfg["depth"]),
            d_model=self.embed_dim,
            nhead=int(encoder_cfg["nhead"]),
            dim_feedforward=int(encoder_cfg["dim_feedforward"]),
            dropout=float(encoder_cfg.get("dropout", 0.0)),
            qkv_bias=bool(encoder_cfg.get("qkv_bias", True)),
        )
        self.topk_layers = min(
            int(cfg.get("target", {}).get("topk_layers", encoder_cfg["depth"])),
            int(encoder_cfg["depth"]),
        )

    def token_lengths(self, waveform_lengths):
        if waveform_lengths is None:
            return None
        values = waveform_lengths.detach().cpu().tolist() if torch.is_tensor(waveform_lengths) else waveform_lengths
        return torch.as_tensor(
            [self.feature_extractor.output_length(int(length)) for length in values],
            dtype=torch.long,
            device=waveform_lengths.device if torch.is_tensor(waveform_lengths) else None,
        )

    def encode_waveform(self, waveforms):
        features = self.feature_extractor(waveforms)
        features = self.feature_norm(features)
        features = self.post_extraction_mapper(features)
        return self.pos_embed_encoder(features)

    def forward(self, waveforms, valid_wave_lens=None, return_all_layers=False):
        tokens = self.encode_waveform(waveforms)
        token_lens = self.token_lengths(valid_wave_lens)
        padding_mask = None
        if token_lens is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
            padding_mask = positions >= token_lens[:, None]
        output, layers = self.context_encoder(
            tokens,
            key_padding_mask=padding_mask,
            return_all_layers=True,
        )
        return (output, layers, padding_mask) if return_all_layers else output

    def extract_features(self, waveforms, valid_wave_lens=None, pool=False):
        _, layers, padding_mask = self.forward(
            waveforms, valid_wave_lens=valid_wave_lens, return_all_layers=True
        )
        features = topk_average(layers, self.topk_layers)
        if not pool:
            return features, padding_mask
        if padding_mask is None:
            return features.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
        return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

