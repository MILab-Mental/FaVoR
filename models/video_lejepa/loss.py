import math

import torch
from torch import nn

from .sigreg import SIGReg


class LeJEPALoss(nn.Module):
    def __init__(self, sigreg_weight=0.02, **sigreg_kwargs):
        super().__init__()
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg = SIGReg(**sigreg_kwargs)

    def forward(self, embeddings):
        """Compute LeVJEPA objective from ``[batch, views, dim]`` embeddings."""
        if embeddings.ndim != 3 or embeddings.shape[1] < 1:
            raise ValueError(f"expected [batch,views,dim], got {tuple(embeddings.shape)}")
        global_embedding = embeddings[:, :1]  # deliberately not detached
        invariance = (global_embedding - embeddings).pow(2).mean()
        sigreg = self.sigreg(embeddings.permute(1, 0, 2))
        return {
            "loss": invariance + self.sigreg_weight * sigreg,
            "invariance_loss": invariance,
            "sigreg_loss": sigreg,
        }


@torch.no_grad()
def embedding_statistics(embeddings):
    flat = embeddings.float().reshape(-1, embeddings.shape[-1])
    centered = flat - flat.mean(0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).mean()
    singular = torch.linalg.svdvals(centered)
    probs = singular / singular.sum().clamp_min(1e-12)
    rankme = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
    return {"embedding_std": std, "rankme": rankme}
