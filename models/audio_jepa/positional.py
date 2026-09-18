import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Fixed 1D sinusoidal positional embedding.
    Compatible with WavJEPA-style non-learnable positional embedding.
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0 ** omega)

    pos = np.arange(length, dtype=np.float64)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    return emb


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pos = get_1d_sincos_pos_embed(embed_dim, max_len)
        pos = torch.from_numpy(pos).float().unsqueeze(0)  # [1, max_len, D]
        self.register_buffer("pos", pos, persistent=False)

    def forward(self, x):
        """
        x: [B, N, D]
        训练时 N 固定等于 max_len（2s 裁剪对应的帧数）；
        但推理时（如完整 utterance 提取 frame-level 特征）N 可能超过 max_len，
        此时现算超出部分的 sincos 位置编码并拼接，而不是直接报错或截断到 max_len。
        """
        N = x.size(1)
        if N <= self.max_len:
            pos = self.pos[:, :N, :]
        else:
            extra = get_1d_sincos_pos_embed(self.embed_dim, N)
            extra = torch.from_numpy(extra).float().unsqueeze(0).to(x.device)
            pos = extra
        return x + pos
    
