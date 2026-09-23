import random
import torch


def _sample_spans(N, prob, length, max_tries=100):
    """
    与官方 time-inverse masker 一致的 span 采样：
    span 数 = int(prob * N / length + random())，每个 span 随机起点、长度 length，
    多个 span 取并集（允许重叠）。
    """
    mask = torch.zeros(N, dtype=torch.bool)
    num_spans = int(prob * N / float(length) + random.random())
    num_spans = max(1, num_spans)

    for _ in range(num_spans):
        start = random.randint(0, max(0, N - length))
        end = min(N, start + length)
        mask[start:end] = True

    return mask


class TimeInverseBlockMasker50Hz:
    """
    对齐官方 config/masker/AudioSet.yaml：
      name: time-inverse
      context_mask_prob: 0.65, context_mask_length: 10
      target_masks_per_context: 4
      target_prob: 0.25, target_length: 10
      ratio_cutoff: 0.1

    context：先按 (context_mask_prob, context_mask_length) 生成 span 并集当作"遮住"的区域，
             取反得到 context 可见位；
    target：K 个 target group，每组独立按 (target_prob, target_length) 生成 span 并集
            （不再是单个不重叠的连续 block，而是和 context 同一套机制）；
    最终把与任一 target 重叠的位置从 context 中剔除，并保证 context 覆盖率 >= ratio_cutoff。
    """

    def __init__(
        self,
        total_patches,
        target_masks_per_context=4,
        target_prob=0.25,
        target_length=10,
        context_mask_prob=0.65,
        context_mask_length=10,
        min_context_ratio=0.10,
        max_tries=100,
        curriculum=False, 
        curriculum_ramp=0.3,
    ):
        self.N = total_patches
        self.K = target_masks_per_context
        self.target_prob = target_prob
        self.target_length = target_length
        self.context_mask_prob = context_mask_prob
        self.context_mask_length = context_mask_length
        self.min_context_ratio = min_context_ratio
        self.max_tries = max_tries
        self.curriculum = curriculum
        self.curriculum_ramp = curriculum_ramp

    def _sample_one(self, N, context_mask_prob):
        for _ in range(self.max_tries):
            # 1. K 个 target group，每组独立 span 采样
            target_positions = torch.stack([
                _sample_spans(N, self.target_prob, self.target_length, self.max_tries)
                for _ in range(self.K)
            ])  # [K, N]
            target_any = target_positions.any(dim=0)

            # 2. context：遮住区域取反
            context_mask = _sample_spans(N, context_mask_prob, self.context_mask_length, self.max_tries)
            visible = ~context_mask
            visible[target_any] = False

            if visible.float().mean().item() >= self.min_context_ratio:
                return visible, target_positions

        # 兜底：与原实现一致，取前 min_context_ratio 比例作为 context，K 个连续 block 作为 target
        visible = torch.zeros(N, dtype=torch.bool)
        visible[:max(1, int(self.min_context_ratio * N))] = True

        target_positions = torch.zeros(self.K, N, dtype=torch.bool)
        for k in range(self.K):
            start = k * self.target_length
            end = min(N, start + self.target_length)
            target_positions[k, start:end] = True
            visible[start:end] = False

        return visible, target_positions

    def __call__(self, batch_size, device, lengths=None, step=None, total_steps=None):
        prob = self.context_mask_prob
        if self.curriculum and step is not None and total_steps:
            progress = min(1.0, step / max(1, total_steps * self.curriculum_ramp))
            prob = prob * (0.6 + 0.4 * progress)

        visibles, targets = [], []
        for i in range(batch_size):
            N_i = int(lengths[i]) if lengths is not None else self.N
            v, t = self._sample_one(N_i, prob)
            visibles.append(v)
            targets.append(t)

        Nmax = max(v.size(0) for v in visibles)
        vis = torch.zeros(batch_size, Nmax, dtype=torch.bool)
        tgt = torch.zeros(batch_size, self.K, Nmax, dtype=torch.bool)
        for i, (v, t) in enumerate(zip(visibles, targets)):
            vis[i, :v.size(0)] = v
            tgt[i, :, :t.size(1)] = t
        return vis.to(device), tgt.to(device)

    # def __call__(self, batch_size, device):
    #     context_visible_list = []
    #     target_positions_list = []

    #     for _ in range(batch_size):
    #         visible, target_positions = self._sample_one()
    #         context_visible_list.append(visible)
    #         target_positions_list.append(target_positions)

    #     context_visible = torch.stack(context_visible_list, dim=0).to(device)
    #     target_positions = torch.stack(target_positions_list, dim=0).to(device)

    #     return context_visible, target_positions
