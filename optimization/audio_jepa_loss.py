import torch
import torch.nn.functional as F


def jepa_loss(preds, targets, target_positions):
    """
    preds: [B*K, N, D]
    targets: [B, N, D]
    target_positions: [B, K, N], bool

    Compute MSE only on target positions.
    """
    B, K, N = target_positions.shape
    D = preds.shape[-1]

    preds = preds.view(B, K, N, D)
    targets = targets.unsqueeze(1).repeat(1, K, 1, 1)  # [B, K, N, D]

    loss = F.mse_loss(preds, targets, reduction="none")  # [B, K, N, D]
    loss = loss.mean(dim=-1)                             # [B, K, N]

    mask = target_positions.float()
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

    return loss
