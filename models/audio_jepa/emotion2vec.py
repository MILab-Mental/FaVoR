"""Self-contained emotion2vec audio front-end components.

These modules reproduce the parameterized inference path of the official
data2vec_multi audio encoder without requiring fairseq at runtime.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RelativePositionEncoder(nn.Module):
    """Five grouped Conv1d blocks used by emotion2vec+ large."""

    def __init__(self, embed_dim, depth=5, width=95, groups=16):
        super().__init__()
        kernel_size = max(3, int(width) // int(depth))
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "conv": nn.Conv1d(
                    embed_dim, embed_dim, kernel_size,
                    padding=kernel_size // 2, groups=groups,
                ),
                "norm": nn.LayerNorm(embed_dim, elementwise_affine=False),
                "activation": nn.GELU(),
            }) for _ in range(depth)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer["conv"](x)
            # fairseq SamePad removes the final position for even kernels.
            if self.kernel_size % 2 == 0:
                x = x[..., :-1]
            x = layer["norm"](x.transpose(1, 2)).transpose(1, 2)
            x = layer["activation"](x)
        return x.transpose(1, 2)


def alibi_bias(batch_size, time_steps, heads, *, dtype, device):
    """Symmetric non-causal ALiBi used by the official audio encoder."""

    def power_of_two_slopes(count):
        start = 2 ** (-(2 ** -(math.log2(count) - 3)))
        return [start * start ** index for index in range(count)]

    if math.log2(heads).is_integer():
        slopes = power_of_two_slopes(heads)
    else:
        closest = 2 ** math.floor(math.log2(heads))
        slopes = power_of_two_slopes(closest)
        slopes += power_of_two_slopes(2 * closest)[0::2][:heads - closest]
    slopes = torch.tensor(slopes, dtype=dtype, device=device)
    positions = torch.arange(time_steps, device=device)
    distances = -(positions[:, None] - positions[None, :]).abs()
    bias = slopes[:, None, None] * distances.to(dtype)[None]
    return bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
