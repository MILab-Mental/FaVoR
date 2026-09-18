import torch
import torch.nn.functional as F


def topk_average(layer_outputs, topk_layers=8):
    """
    与 topk_instance_average 的区别：不对每层做 instance_norm（不强制沿时间维零均值），
    只是简单地取最后 k 层输出并在层维度上取平均。
    用于下游特征提取（linear probe / MLP probe / UMAP 等），保留样本间可区分的统计信息；
    topk_instance_average 仅用于 JEPA 训练时构造 target，不要在特征提取路径里复用。
    """
    k = min(topk_layers, len(layer_outputs))
    top_layers = layer_outputs[-k:]
    stacked = torch.stack(top_layers, dim=0)  # [k, B, N, D]
    return stacked.mean(dim=0)                # [B, N, D]


def topk_instance_average(layer_outputs, topk_layers=8):
    """
    WavJEPA / emotion2vec style target construction.

    layer_outputs: List[Tensor]
        Each tensor has shape [B, N, D].

    For each of top-k layers:
      1. instance normalize over time dimension
      2. then average across layers
    """
    k = min(topk_layers, len(layer_outputs))
    top_layers = layer_outputs[-k:]

    normed = []
    for x in top_layers:
        # x: [B, N, D]
        x = x.transpose(1, 2)          # [B, D, N]
        x = F.instance_norm(x)
        x = x.transpose(1, 2)          # [B, N, D]
        normed.append(x)

    stacked = torch.stack(normed, dim=0)  # [k, B, N, D]
    return stacked.mean(dim=0)            # [B, N, D]


@torch.no_grad()
def ema_update(target_encoder, context_encoder, tau):
    """
    EMA update target encoder from context encoder.
    """
    for p_t, p_s in zip(target_encoder.parameters(), context_encoder.parameters()):
        p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)


def tau_schedule(step, tau_start=0.999, tau_end=0.99999, anneal_steps=100000):
    if step >= anneal_steps:
        return tau_end
    return tau_start + (tau_end - tau_start) * step / anneal_steps
