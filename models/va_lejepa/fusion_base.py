"""F1 fusion; modality adapters, sequence construction and attention stay separate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

BRIDGE_REGISTRY = {'concat_transformer'}


def sincos(positions, dim, dtype):
    frequencies = torch.exp(torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) * (-math.log(10000) / dim))
    angles = positions.float()[..., None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)[..., :dim].to(dtype)


class ModalityInputAdapters(nn.Module):
    def __init__(self, video_dim, audio_dim, common_dim, hidden_dim):
        super().__init__()
        def adapter(dim):
            return nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, common_dim), nn.LayerNorm(common_dim))
        self.video = adapter(video_dim)
        self.audio = adapter(audio_dim)


class ConcatTransformerFusion(nn.Module):
    def __init__(self, mode, common_dim, cfg):
        super().__init__()
        self.mode, self.dim = mode, common_dim
        self.max_video = int(cfg.get('max_video_tokens_per_time', 16))
        self.max_audio = int(cfg.get('max_audio_tokens_per_time', 4))
        if min(self.max_video, self.max_audio) < 1:
            raise ValueError('EarlyFusion token budgets must be positive')
        if cfg.get('budget_strategy', 'pooling') != 'pooling':
            raise ValueError('v1 supports deterministic token pooling only')
        self.modality = nn.Parameter(torch.randn(3, common_dim) * .02)
        self.fuse = nn.Parameter(torch.randn(1, 1, common_dim) * .02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(common_dim, int(cfg.get('nhead', 12)),
            dim_feedforward=int(cfg.get('dim_feedforward', 4 * common_dim)),
            dropout=float(cfg.get('dropout', 0)), batch_first=True, norm_first=True)
            for _ in range(int(cfg.get('depth', 4)))])
        if not self.layers:
            raise ValueError('fusion depth must be positive')
        self.norm = nn.LayerNorm(common_dim)
        self.checkpointing = bool(cfg.get('gradient_checkpointing', True))

    def build_sequence(self, aligned, adapters):
        video, audio = aligned.video, aligned.audio
        batch, temporal, spatial, _ = video.tokens.shape
        # Bin centers, in seconds relative to this paired clip, are shared by A/V.
        positions = (aligned.bin_edges[:, 1:] + aligned.bin_edges[:, :-1]) / 2 - aligned.bin_edges[:, :1]
        time = sincos(positions, self.dim, video.tokens.dtype)
        if self.mode == 'late':
            v = adapters.video(video.pooled) + self.modality[0]
            a = adapters.audio(audio.pooled) + self.modality[1]
            seq = torch.stack((v, a), 1)
            mask = torch.stack((video.padding_mask.all(1), audio.padding_mask.all(1)), 1)
        elif self.mode == 'feature':
            v = adapters.video(video.tokens.mean(2)) + self.modality[0] + time
            a = adapters.audio(aligned.audio_features) + self.modality[1] + time
            seq = torch.stack((v, a), 2).flatten(1, 2)
            mask = torch.stack((video.padding_mask, aligned.audio_padding_mask), 2).flatten(1, 2)
        elif self.mode == 'early':
            nv = min(spatial, self.max_video)
            # Pool within each spatial grid, never across time.
            vt = F.adaptive_avg_pool1d(video.tokens.reshape(batch * temporal, spatial, -1).transpose(1, 2), nv)
            vt = vt.transpose(1, 2).reshape(batch, temporal, nv, -1)
            v = adapters.video(vt) + self.modality[0] + time[:, :, None]
            v = v + sincos(torch.arange(nv, device=v.device), self.dim, v.dtype)[None, None]
            # Token pooling is bounded to tokens assigned by physical time to this bin.
            audio_bins, masks = [], []
            for b in range(batch):
                bins, bin_masks = [], []
                for t in range(temporal):
                    selected = audio.tokens[b, aligned.assignment[b] == t]
                    count = min(selected.shape[0], self.max_audio)
                    if count:
                        pooled = F.adaptive_avg_pool1d(selected.T[None], count)[0].T
                        pooled = F.pad(pooled, (0, 0, 0, self.max_audio - count))
                    else:
                        pooled = audio.tokens.new_zeros(self.max_audio, audio.tokens.shape[-1])
                    bins.append(pooled)
                    bin_masks.append(torch.arange(self.max_audio, device=v.device) >= count)
                audio_bins.append(torch.stack(bins))
                masks.append(torch.stack(bin_masks))
            a = adapters.audio(torch.stack(audio_bins)) + self.modality[1] + time[:, :, None]
            seq = torch.cat((v, a), 2).flatten(1, 2)
            mask = torch.cat((video.padding_mask[:, :, None].expand(-1, -1, nv), torch.stack(masks)), 2).flatten(1, 2)
        else:
            raise ValueError(f'unknown fusion mode {self.mode}')
        seq = seq.masked_fill(mask[..., None], 0)
        prefix = (self.fuse + self.modality[2]).expand(batch, -1, -1)
        return torch.cat((prefix, seq), 1), F.pad(mask, (1, 0), value=False)

    def forward(self, aligned, adapters):
        seq, mask = self.build_sequence(aligned, adapters)
        for layer in self.layers:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                seq = checkpoint(layer, seq, src_key_padding_mask=mask, use_reentrant=False)
            else:
                seq = layer(seq, src_key_padding_mask=mask)
        return self.norm(seq[:, 0])
