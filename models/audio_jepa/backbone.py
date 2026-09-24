"""Reusable waveform backbone shared by audio pre-training and fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from .emotion2vec import RelativePositionEncoder, alibi_bias
from .extractor import ConvFeatureExtractor
from .target import topk_average
from .transformer import Emotion2VecTransformerEncoder


class AudioBackbone(nn.Module):
    """emotion2vec-compatible CNN and Transformer without JEPA-only modules."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        audio_cfg = cfg["audio"]
        backbone_cfg = cfg.get("backbone", {})
        extractor_cfg = cfg["extractor"]
        encoder_cfg = cfg["encoder"]
        self.mode = str(backbone_cfg.get("mode", "emotion2vec")).lower()
        if self.mode != "emotion2vec":
            raise ValueError("AudioBackbone supports only backbone.mode='emotion2vec'")

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
        self._build_emotion2vec(backbone_cfg, encoder_cfg)
        self.topk_layers = min(
            int(cfg.get("target", {}).get("topk_layers", encoder_cfg["depth"])),
            int(encoder_cfg["depth"]),
        )
        features_cfg = cfg.get("features", {})
        default_feature_mode = "last"
        self.feature_mode = str(features_cfg.get("mode", default_feature_mode)).lower()
        self.feature_topk_layers = int(features_cfg.get("topk_layers", self.topk_layers))
        if self.feature_mode not in {"last", "topk_average"}:
            raise ValueError("model.features.mode must be 'last' or 'topk_average'")
        if self.feature_mode == "topk_average" and self.feature_topk_layers < 1:
            raise ValueError("model.features.topk_layers must be at least 1")

    def _build_emotion2vec(self, backbone_cfg, encoder_cfg):
        self.num_extra_tokens = int(backbone_cfg.get("num_extra_tokens", 10))
        self.num_heads = int(encoder_cfg["nhead"])
        self.relative_positional_encoder = RelativePositionEncoder(
            self.embed_dim,
            depth=int(backbone_cfg.get("conv_pos_depth", 5)),
            width=int(backbone_cfg.get("conv_pos_width", 95)),
            groups=int(backbone_cfg.get("conv_pos_groups", 16)),
        )
        self.extra_tokens = nn.Parameter(torch.zeros(1, self.num_extra_tokens, self.embed_dim))
        if self.num_extra_tokens > 1:
            nn.init.normal_(self.extra_tokens[:, 1:])
        self.alibi_scale = nn.Parameter(
            torch.ones(1, 1, self.num_heads, 1, 1),
            requires_grad=bool(backbone_cfg.get("learned_alibi_scale", False)),
        )
        common = dict(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(encoder_cfg["dim_feedforward"]),
            dropout=float(encoder_cfg.get("dropout", 0.0)),
            attention_dropout=float(encoder_cfg.get("attention_dropout", 0.0)),
            activation_dropout=float(encoder_cfg.get("activation_dropout", 0.0)),
            post_mlp_dropout=float(encoder_cfg.get("post_mlp_dropout", 0.0)),
            qkv_bias=bool(encoder_cfg.get("qkv_bias", True)),
            layer_norm_first=bool(encoder_cfg.get("layer_norm_first", False)),
            ffn_targets=bool(encoder_cfg.get("ffn_targets", True)),
            norm_eps=float(encoder_cfg.get("norm_eps", 1e-6)),
        )
        self.modality_context_encoder = Emotion2VecTransformerEncoder(
            depth=int(backbone_cfg.get("prenet_depth", 4)),
            input_norm=not common["layer_norm_first"],
            input_dropout=float(backbone_cfg.get("prenet_dropout", 0.0)),
            **common,
        )
        self.context_encoder = Emotion2VecTransformerEncoder(
            depth=int(encoder_cfg["depth"]),
            final_norm=common["layer_norm_first"],
            input_dropout=float(encoder_cfg.get("dropout_input", 0.0)),
            **common,
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

    def _project_waveform(self, waveforms):
        features = self.feature_extractor(waveforms)
        features = self.feature_norm(features)
        features = self.post_extraction_mapper(features)
        return features

    def prepare_transformer_input(self, waveforms, valid_wave_lens=None):
        """Build the input expected by the shared/global Transformer blocks."""
        tokens = self._project_waveform(waveforms)
        token_lens = self.token_lengths(valid_wave_lens)
        padding_mask = None
        if token_lens is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
            padding_mask = positions >= token_lens[:, None]
        tokens = tokens + self.relative_positional_encoder(tokens)
        bias = alibi_bias(
            tokens.shape[0], tokens.shape[1], self.num_heads,
            dtype=torch.float32, device=tokens.device,
        )
        scale = self.alibi_scale.clamp_min(0)
        if scale.shape[0] == 1:
            bias = bias * scale.squeeze(0).to(bias.dtype)
            scale = None
        if self.num_extra_tokens:
            tokens = torch.cat((self.extra_tokens.expand(tokens.shape[0], -1, -1), tokens), dim=1)
            if padding_mask is not None:
                padding_mask = torch.nn.functional.pad(padding_mask, (self.num_extra_tokens, 0))
            bias = torch.nn.functional.pad(
                bias, (self.num_extra_tokens, 0, self.num_extra_tokens, 0)
            )
        tokens = self.modality_context_encoder(
            tokens, key_padding_mask=padding_mask, alibi_bias=bias,
            alibi_scale=scale, return_all_layers=False,
        )
        return tokens, padding_mask, bias, scale

    def encode_waveform(self, waveforms, valid_wave_lens=None):
        """Backward-compatible alias returning the global encoder input."""
        return self.prepare_transformer_input(waveforms, valid_wave_lens)[0]

    def encode_transformer(self, tokens, padding_mask=None, alibi_bias_value=None,
                           alibi_scale=None, return_all_layers=False,
                           remove_extra_tokens=True):
        encoded = self.context_encoder(
            tokens, key_padding_mask=padding_mask,
            alibi_bias=alibi_bias_value, alibi_scale=alibi_scale,
            return_all_layers=return_all_layers,
        )
        if return_all_layers:
            output, layers = encoded
        else:
            output, layers = encoded, None
        if remove_extra_tokens and self.num_extra_tokens:
            output = output[:, self.num_extra_tokens:]
            if layers is not None:
                layers = [layer[:, self.num_extra_tokens:] for layer in layers]
            if padding_mask is not None:
                padding_mask = padding_mask[:, self.num_extra_tokens:]
        return (output, layers, padding_mask) if return_all_layers else output

    def forward(self, waveforms, valid_wave_lens=None, return_all_layers=False):
        tokens, padding_mask, bias, scale = self.prepare_transformer_input(
            waveforms, valid_wave_lens
        )
        if not return_all_layers:
            return self.encode_transformer(
                tokens, padding_mask, bias, scale,
                return_all_layers=False, remove_extra_tokens=True,
            )
        output, layers, padding_mask = self.encode_transformer(
            tokens, padding_mask, bias, scale,
            return_all_layers=True, remove_extra_tokens=True,
        )
        return output, layers, padding_mask

    def extract_features(self, waveforms, valid_wave_lens=None, pool=False):
        if self.feature_mode == "last":
            tokens, padding_mask, bias, scale = self.prepare_transformer_input(
                waveforms, valid_wave_lens
            )
            features = self.encode_transformer(
                tokens, padding_mask, bias, scale,
                return_all_layers=False, remove_extra_tokens=True,
            )
            if padding_mask is not None and self.num_extra_tokens:
                padding_mask = padding_mask[:, self.num_extra_tokens:]
        else:
            _, layers, padding_mask = self.forward(
                waveforms, valid_wave_lens=valid_wave_lens, return_all_layers=True
            )
            features = topk_average(layers, self.feature_topk_layers)
        if not pool:
            return features, padding_mask
        if padding_mask is None:
            return features.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
        return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
