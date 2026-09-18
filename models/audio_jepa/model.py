"""Audio JEPA pre-training container built around :class:`AudioBackbone`."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .backbone import AudioBackbone
from datasets.masks.audio_jepa import TimeInverseBlockMasker50Hz
from optimization.audio_jepa_loss import jepa_loss
from .predictor import JEPAPredictor
from .target import topk_instance_average


class AudioJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AudioBackbone(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

        predictor_cfg = cfg["predictor"]
        self.predictor = JEPAPredictor(
            encoder_dim=self.encoder.embed_dim,
            decoder_dim=int(predictor_cfg["d_model"]),
            depth=int(predictor_cfg["depth"]),
            nhead=int(predictor_cfg["nhead"]),
            dim_feedforward=int(predictor_cfg["dim_feedforward"]),
            total_patches=self.encoder.max_tokens,
            dropout=float(predictor_cfg.get("dropout", 0.0)),
        )
        mask_cfg = cfg["mask"]
        self.masker = TimeInverseBlockMasker50Hz(
            total_patches=self.encoder.max_tokens,
            target_masks_per_context=int(mask_cfg["target_masks_per_context"]),
            target_prob=float(mask_cfg["target_prob"]),
            target_length=int(mask_cfg["target_length"]),
            context_mask_prob=float(mask_cfg["context_mask_prob"]),
            context_mask_length=int(mask_cfg["context_mask_length"]),
            min_context_ratio=float(mask_cfg["min_context_ratio"]),
            max_tries=int(mask_cfg.get("max_tries", 100)),
            curriculum=bool(mask_cfg.get("curriculum", False)),
            curriculum_ramp=float(mask_cfg.get("curriculum_ramp", 0.3)),
        )

    def sync_target_encoder(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, waveforms, valid_wave_lens=None, step=None, total_steps=None):
        embedded = self.encoder.encode_waveform(waveforms)
        batch_size, token_count = embedded.shape[:2]
        token_lens = self.encoder.token_lengths(valid_wave_lens)
        if token_lens is None:
            pad_mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=embedded.device)
        else:
            pad_mask = torch.arange(token_count, device=embedded.device)[None] >= token_lens[:, None]

        context_visible, target_positions = self.masker(
            batch_size, embedded.device, lengths=token_lens, step=step, total_steps=total_steps
        )
        context_mask = (~context_visible) | pad_mask
        target_positions = target_positions & ~pad_mask[:, None]
        context_last, _ = self.encoder.context_encoder(
            embedded, key_padding_mask=context_mask, return_all_layers=True
        )
        with torch.no_grad():
            _, target_layers = self.target_encoder.context_encoder(
                embedded, key_padding_mask=pad_mask, return_all_layers=True
            )
            targets = topk_instance_average(target_layers, self.encoder.topk_layers)
        predictions = self.predictor(context_last, context_mask, target_positions)
        loss = jepa_loss(predictions, targets, target_positions)
        return loss, {
            "loss": loss.detach(),
            "context_ratio": (context_visible & ~pad_mask).float().sum() / (~pad_mask).float().sum().clamp(min=1),
            "target_ratio": target_positions.float().sum() / ((~pad_mask).sum() * target_positions.shape[1]).clamp(min=1),
            "target_std": targets.detach().std(),
            "pad_ratio": pad_mask.float().mean(),
        }
