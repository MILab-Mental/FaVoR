import torch
import torch.distributed as dist
from torch import nn


class _DistributedAverage(torch.autograd.Function):
    """All-reduce an average while preserving the local autograd path.

    DDP later averages parameter gradients across ranks, so backward must not
    perform another collective or world-size scaling.
    """

    @staticmethod
    def forward(ctx, value):
        if dist.is_available() and dist.is_initialized():
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(dist.get_world_size())
        return value

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from LeVJEPA.

    Input is ``[num_views, batch, projection_dim]``.  The empirical
    characteristic function is averaged over the global DDP batch.
    """

    def __init__(self, knots=17, num_proj=1024, normalize_by_n=False):
        super().__init__()
        if knots < 2:
            raise ValueError("SIGReg knots must be at least 2")
        self.num_proj = int(num_proj)
        self.normalize_by_n = bool(normalize_by_n)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects [views,batch,dim], got {tuple(proj.shape)}")
        directions = torch.randn(
            proj.shape[-1], self.num_proj, device=proj.device, dtype=proj.dtype
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
        # ECF coordinates must be identical across ranks.  Per-rank data
        # loading and augmentation advance RNG streams differently, so relying
        # on equal initial seeds is not sufficient.
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(directions, src=0)
        x_t = (proj @ directions).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        ecf = torch.stack((x_t.cos().mean(-3), x_t.sin().mean(-3)))
        ecf = _DistributedAverage.apply(ecf)
        error = (ecf[0] - self.phi.to(dtype=proj.dtype)).square() + ecf[1].square()
        statistic = error @ self.weights.to(dtype=proj.dtype)
        if not self.normalize_by_n:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            statistic = statistic * proj.shape[-2] * world_size
        return statistic.mean()
