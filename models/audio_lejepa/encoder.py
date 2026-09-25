"""AUDIO-LeJEPA encoder adapter over FAVOR's emotion2vec backbone."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from models.audio_jepa.backbone import AudioBackbone
from utils.audio_checkpoint import load_official_emotion2vec_backbone

LOGGER = logging.getLogger(__name__)


@dataclass
class AudioTokenOutput:
    embedding: Tensor
    tokens: Tensor
    padding_mask: Tensor
    extra_tokens: Tensor
    token_lengths: Tensor


class AudioLeEncoder(nn.Module):
    """Expose extra-token and mask-aware pooling without changing AudioBackbone."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = AudioBackbone(cfg)
        pooling_cfg = cfg.get("pooling", {})
        self.default_pooling = str(pooling_cfg.get("type", "extra_token"))
        self.extra_token_index = int(pooling_cfg.get("index", 0))
        if not 0 <= self.extra_token_index < self.backbone.num_extra_tokens:
            raise ValueError(
                f"pooling.index={self.extra_token_index} outside "
                f"[0,{self.backbone.num_extra_tokens})"
            )

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    def forward(self, waveforms, lengths=None, return_tokens=False, pooling=None):
        pooling = str(pooling or self.default_pooling).lower()
        prepared, transformer_mask, bias, scale = self.backbone.prepare_transformer_input(
            waveforms, lengths
        )
        encoded = self.backbone.encode_transformer(
            prepared,
            transformer_mask,
            bias,
            scale,
            return_all_layers=False,
            remove_extra_tokens=False,
        )
        count = self.backbone.num_extra_tokens
        extra_tokens = encoded[:, :count]
        tokens = encoded[:, count:]
        if transformer_mask is None:
            padding_mask = torch.zeros(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
            token_lengths = torch.full(
                (tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device
            )
        else:
            padding_mask = transformer_mask[:, count:]
            token_lengths = (~padding_mask).sum(1)

        if pooling == "extra_token":
            embedding = extra_tokens[:, self.extra_token_index]
        elif pooling == "masked_mean":
            valid = (~padding_mask).unsqueeze(-1).to(tokens.dtype)
            embedding = (tokens * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            raise ValueError("pooling must be 'extra_token' or 'masked_mean'")
        if not return_tokens:
            return embedding
        return AudioTokenOutput(
            embedding=embedding,
            tokens=tokens,
            padding_mask=padding_mask,
            extra_tokens=extra_tokens,
            token_lengths=token_lengths,
        )


def load_emotion2vec_encoder(encoder, checkpoint_path, min_load_ratio=0.95):
    """Load only a raw official emotion2vec backbone, with an audited report."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(checkpoint, dict) and any(
        key in checkpoint for key in ("target_encoder", "predictor", "optimizer", "ema")
    ):
        raise ValueError(
            "AUDIO-LeJEPA initialization accepts only the raw emotion2vec checkpoint; "
            "AudioJEPA/AEmo-JEPA training checkpoints are not supported"
        )
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("raw emotion2vec checkpoint does not contain a model state mapping")
    if not any(key.startswith("d2v_model.blocks.") for key in state):
        raise ValueError(
            "checkpoint is not a raw emotion2vec/data2vec model state "
            "(missing d2v_model.blocks.*)"
        )
    report = load_official_emotion2vec_backbone(
        encoder.backbone, state, min_parameter_ratio=float(min_load_ratio)
    )
    extra_loaded = "extra_tokens" not in report["missing_targets"]
    report["extra_tokens_loaded"] = extra_loaded
    if extra_loaded:
        LOGGER.info("emotion2vec extra_tokens were loaded from %s", checkpoint_path)
    else:
        LOGGER.warning("emotion2vec extra_tokens are randomly initialized")
    LOGGER.info(
        "AUDIO-LeJEPA initialization report: loaded=%.2f%% missing=%d skipped=%d extra_tokens=%s",
        100 * report["target_ratio"], len(report["missing_targets"]),
        len(report["skipped_sources"]), "loaded" if extra_loaded else "random",
    )
    return report
