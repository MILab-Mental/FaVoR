import torch
import torch.nn as nn


class LayerWeightedRegressor(nn.Module):
    """Learn shared layer weights for a regression task from pooled clip features."""

    def __init__(self, embed_dim, num_layers, dropout=0.0, hidden_dim=256):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.output = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, layer_features):
        weights = self.layer_logits.softmax(dim=0)
        return self.output((layer_features * weights[:, None]).sum(dim=1))
