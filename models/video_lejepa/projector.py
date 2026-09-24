import torch
from torch import nn


class Projector(nn.Module):
    """LeVJEPA's two-layer projection MLP, shape-preserving before the last dim."""

    def __init__(self, input_dim, hidden_dim=2048, output_dim=256, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim) if norm_layer is not None else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        shape = x.shape
        projected = self.net(x.reshape(-1, shape[-1]))
        return projected.reshape(*shape[:-1], projected.shape[-1])
