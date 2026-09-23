# Copyright (c) 2024-present. Pretraining checkpoint representation-quality metrics.
#
# Two metrics, both computed ONLY on user-selected evaluation events so that
# different checkpoints are compared against a *fixed* data subset / *fixed*
# cached batch:
#
#   RankMe  -- effective rank of the (mean) representation matrix Z=[N, D]
#              extracted from the online encoder (global pooled features,
#              prediction heads excluded).  RankMe = exp(-sum p_i log p_i)
#              with p_i = sigma_i / sum_j sigma_j from an SVD of the column
#              mean-centered Z.  Aux stats: per-dim variance and the fraction
#              of total energy held by the top singular value (rank-collapse
#              indicator).  Optionally also averages a per-clip token RankMe.
#
#   Hessian -- Hutchinson trace estimate of the training-loss Hessian through
#              Hessian-vector products.  No explicit Hessian is materialised:
#                 g  = grad_theta L                       (create_graph=True)
#                 Hv = grad_theta (g^T v)  for Rademacher v ~ {-1,+1}^P
#                 Tr(H) ~= mean_v (v^T Hv)
#              computed on a tiny FIXED cached batch (clips + masks generated
#              once by a private MaskCollator) so the trace is comparable
#              across checkpoints and across process restarts.
#
# Distributed notes: RankMe is computed by every rank over its own fixed slice
# of the subset, then all_gather -> every rank sees the same [N,D] matrix
# (cheap SVD, so it is fine to repeat).  The Hessian pass is done on rank 0
# only; other ranks join a barrier afterwards.  Feature / HVP passes use the
# bare `.module` (not the DDP wrappers) to avoid triggering DDP hooks, and
# activation checkpointing is temporarily disabled because autograd
# double-backward is not compatible with it.
from __future__ import annotations

import contextlib
import os
import random
import typing as T

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.video_jepa.masks.collator import MaskCollator
from models.video_jepa.utils.masking import apply_masks

__all__ = ["PretrainEvalMetrics", "build_center_crop_transform", "rankme_metrics"]

_LOG_EPS = 1.0e-12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _dist():
    enabled = dist.is_available() and dist.is_initialized()
    return (
        dist.get_rank() if enabled else 0,
        dist.get_world_size() if enabled else 1,
    )


def _unwrap(module):
    # DDP wrapper or torch.compile wrapper -> the real nn.Module underneath.
    while hasattr(module, "module") and not isinstance(module, torch.nn.ModuleList):
        module = module.module
    return module


def _all_gather_rows(z_local: torch.Tensor):
    """Concatenate rows of `z_local` from every rank.  Equal row counts required."""
    rank, world = _dist()
    if world <= 1:
        return z_local
    z_local = z_local.contiguous()
    gathered = [torch.empty_like(z_local) for _ in range(world)]
    dist.all_gather(gathered, z_local)
    return torch.cat(gathered, dim=0)


def _mean_reduce_tensor(x: torch.Tensor):
    """Average a scalar/1d tensor over all ranks (all ranks call this)."""
    rank, world = _dist()
    if world <= 1:
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x / world


@contextlib.contextmanager
def _activation_checkpointing_disabled(*modules):
    """Temporarily disable activation checkpointing (incompatible w/ 2nd order)."""
    holder = {}
    for m in modules:
        if m is None:
            continue
        m = _unwrap(m)
        for name in ("use_activation_checkpointing",):
            if hasattr(m, name):
                holder[id(m)] = (m, getattr(m, name))
                setattr(m, name, False)
    try:
        yield
    finally:
        for m, old in holder.values():
            setattr(m, "use_activation_checkpointing", old)


@contextlib.contextmanager
def _force_math_sdpa():
    """While active, every F.scaled_dot_product_attention call that the attention
    modules route through `torch.backends.cuda.sdp_kernel()` runs on the **math**
    backend.  Flash / memory-efficient attention kernels have no double-backward
    implemented, which the Hutchinson HVP pass requires; the math kernel is fully
    higher-order differentiable.  Self-restoring on exit.

    The modules enter an *empty* `with torch.backends.cuda.sdp_kernel():` around
    each SDPA call (see models/utils/modules.py), which would otherwise reset the
    backend flags to default.  Swapping the symbol for a context class that forces
    math per-entry guarantees every individual call selects the math kernel.
    """
    try:
        backends = torch.backends.cuda
        original = backends.sdp_kernel
    except Exception:
        yield
        return

    class _MathOnlySDPA:
        def __enter__(self):
            backends.enable_flash_sdp(False)
            backends.enable_mem_efficient_sdp(False)
            backends.enable_math_sdp(True)

        def __exit__(self, *exc):
            backends.enable_flash_sdp(True)
            backends.enable_mem_efficient_sdp(True)
            backends.enable_math_sdp(True)
            return False

    backends.sdp_kernel = _MathOnlySDPA
    try:
        yield
    finally:
        backends.sdp_kernel = original


@contextlib.contextmanager
def _eval_mode(*modules):
    was_training = []
    for m in modules:
        if m is not None:
            m = _unwrap(m)
            was_training.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, flag in was_training:
            m.train(flag)


# --------------------------------------------------------------------------- #
# deterministic eval clip transform
# --------------------------------------------------------------------------- #
class _CenterCropTransform(object):
    """Deterministic cover-resize + center crop + ImageNet norm.

    Defined at module level (NOT a closure) so that the DataLoader's `spawn`
    workers can pickle it when `num_workers > 0`.  Operates on a clip laid out
    as [T, H, W, C] in [0, 255] (the layout produced by
    VideoDataset.loadvideo_decord) and returns a [C, T, crop, crop] float
    tensor.  It contains no RNG, so every checkpoint sees bit-identical inputs
    for a given video.
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1, 1) * 255.0
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1, 1) * 255.0

    def __call__(self, clip):
        x = torch.as_tensor(clip, dtype=torch.float32)
        if x.ndim != 4 or x.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unexpected clip layout {tuple(x.shape)} (expected T,H,W,C)")
        x = x.permute(3, 0, 1, 2)  # C, T, H, W
        _, _, H, W = x.shape
        if H != self.crop_size or W != self.crop_size:
            scale = max(self.crop_size / H, self.crop_size / W)
            x = F.interpolate(
                x.unsqueeze(0), scale_factor=(1.0, scale, scale), mode="trilinear", align_corners=False
            ).squeeze(0)
            _, _, H, W = x.shape
            y0 = max(0, (H - self.crop_size) // 2)
            x0 = max(0, (W - self.crop_size) // 2)
            x = x[:, :, y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
        return (x - self.mean) / self.std


def build_center_crop_transform(crop_size: int):
    """Factory for a picklable deterministic center-crop transform."""
    return _CenterCropTransform(crop_size)


def _rankme_collate(batch):
    """Module-level collate (picklable): item is ([clip], label, clip_indices)."""
    clips = torch.stack([s[0][0] for s in batch], dim=0)
    return [clips]


def _rankme_worker_init(worker_id):
    """Module-level worker seeder (picklable).  Deterministic so that even the
    rare corrupt-video retry path in VideoDataset.__getitem__ stays reproducible
    across checkpoints."""
    seed = 20240 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _copy_dataset_deterministic(source, transform):
    """Clone a built VideoDataset, sharing its manifest arrays, but force a
    deterministic (fixed clip window + no-RNG transform) reader."""
    # Build an instance of the *same* class without running __init__ (avoids a
    # second 2.7M-row CSV parse), then copy state and swap the stochastic bits.
    clone = source.__class__.__new__(source.__class__)
    clone.__dict__.update(source.__dict__)
    clone.transform = transform
    clone.shared_transform = None  # strip any stochastic/aug closure (unpicklable + RNG)
    clone.random_clip_sampling = False
    clone.motion_shift = False
    return clone


# --------------------------------------------------------------------------- #
# RankMe math
# --------------------------------------------------------------------------- #
def rankme_metrics(Z: torch.Tensor) -> T.Dict[str, float]:
    """RankMe + auxiliary statistics of a (already column mean-centered) Z [N,D].

    Returns dict with:
      rankme_global  - exp(-sum p_i log p_i)
      var_global     - mean per-feature sample variance (= total energy / D)
      top_sv_energy  - sigma_1^2 / sum sigma_i^2   (1.0 -> fully rank-collapsed)
    SVD is run in fp64 on CPU for stability regardless of the source dtype.
    """
    Z = Z.detach().double().cpu()
    if Z.ndim == 1:
        Z = Z.unsqueeze(0)
    n, d = Z.shape
    if n == 0 or d == 0:
        return {"rankme_global": float("nan"), "var_global": float("nan"), "top_sv_energy": float("nan")}
    var_global = float(Z.var(dim=0).mean().clamp_min(0.0))
    sv = torch.linalg.svdvals(Z)  # descending
    s2 = (sv * sv).clamp_min(0.0)
    total = float(s2.sum().clamp_min(_LOG_EPS))
    p = (s2 / total).clamp_min(_LOG_EPS)
    rankme = float(torch.exp(-(p * p.log()).sum()))
    top_sv_energy = float((s2[0] / total).clamp(max=1.0)) if s2.numel() else float("nan")
    return {"rankme_global": rankme, "var_global": var_global, "top_sv_energy": top_sv_energy}


def _per_clip_token_rankme(token_feats: torch.Tensor) -> torch.Tensor:
    """Effective rank of one clip's [n_tok, D] token feature matrix."""
    t = token_feats.detach().double().cpu()
    t = t - t.mean(dim=0)
    sv = torch.linalg.svdvals(t)
    s2 = (sv * sv).clamp_min(0.0)
    total = float(s2.sum().clamp_min(_LOG_EPS))
    p = (s2 / total).clamp_min(_LOG_EPS)
    return float(torch.exp(-(p * p.log()).sum()))


# --------------------------------------------------------------------------- #
# Hutchinson Hessian trace
# --------------------------------------------------------------------------- #
def hutchinson_trace(loss, params, n_vectors: int = 5, base_seed: int = 0) -> T.Tuple[float, float]:
    """Estimate Tr(Hessian) of `loss` w.r.t. `params` via HVP Hutchinson.

    Returns (mean, std) over `n_vectors` Rademacher samples.  `loss` must be a
    differentiable scalar and the forward graph must have been built with
    requires-grad leaves (create_graph inside this function).

    `allow_unused=True` + skip-None: some params legitimately receive no gradient
    from the JEPA loss (e.g. the predictor's `predictor_pos_embed` when
    `use_rope=True`, or unused mask-token embeddings).  Their Hessian rows/columns
    are identically zero, so they contribute nothing to v^T H v and are skipped.
    """
    if n_vectors <= 0 or not params:
        return float("nan"), float("nan")
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    estimates = []
    for itr in range(n_vectors):
        # Rademacher vectors drawn on CPU (torch.randint rejects a CPU generator
        # with a CUDA output device) then moved to each parameter's device.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(base_seed + 7 * itr)
        v = []
        for p in params:
            r = torch.randint(0, 2, p.shape, generator=gen, dtype=torch.float32)
            v.append(r.mul_(2.0).sub_(1.0).to(p.device))
        # dot(g, v)  -- skip params with no gradient
        dot = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for gp, vp in zip(grads, v):
            if gp is not None:
                dot = dot + (gp * vp).sum()
        # Hv = grad_theta (g^T v)  ==  H v
        hv = torch.autograd.grad(dot, params, retain_graph=(itr < n_vectors - 1), allow_unused=True)
        est = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for hvp, vp in zip(hv, v):
            if hvp is not None:
                est = est + (hvp * vp).sum()
        estimates.append(float(est.detach()))
        del v, dot, hv, est
    ests = torch.tensor(estimates, dtype=torch.float64)
    return float(ests.mean()), float(ests.std())


def _pretrain_loss_on_sample(encoder, predictor, target_encoder, clips, masks_enc, masks_pred, loss_exp):
    """Faithful copy of app/pretrain_v/train.py forward_context/forward_target/loss_fn.

    `encoder/predictor/target_encoder` are the bare (unwrapped) modules.
    `clips`      : list (per dataset fpc) of [B, C, T, H, W] tensors
    `masks_enc`  : list (per dataset) of list (per mask generator) of [B, Kctx] ints
    `masks_pred` : list (per dataset) of list (per mask generator) of [B, Kpred] ints
    """
    with torch.no_grad():
        h = target_encoder(clips)
        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
    z = encoder(clips, masks_enc)
    z = predictor(z, masks_enc, masks_pred)

    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    loss, n = 0.0, 0
    for zi, hi in zip(z, h):
        for zij, hij in zip(zi, hi):
            loss = loss + torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    return loss / n


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class PretrainEvalMetrics(object):
    """Runs RankMe / Hessian trace over a fixed evaluation subset of the
    pretraining data, on-demand at checkpoint evaluation events."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device,
        train_dataset,
        cfgs_mask,
        dataset_fpcs,
        crop_size: int,
        patch_size: int,
        tubelet_size: int,
        encoder,
        predictor,
        target_encoder,
        eval_freq: int = -1,
        rankme_cfg: T.Optional[dict] = None,
        hessian_cfg: T.Optional[dict] = None,
        seed: int = 0,
        loss_exp: float = 1.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.train_dataset = train_dataset
        self.cfgs_mask = cfgs_mask
        self.dataset_fpcs = list(dataset_fpcs)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.encoder = _unwrap(encoder)
        self.predictor = _unwrap(predictor)
        self.target_encoder = _unwrap(target_encoder)
        self.eval_freq = int(eval_freq)
        self.seed = int(seed)

        rankme_cfg = rankme_cfg or {}
        hessian_cfg = hessian_cfg or {}

        # -- RankMe
        self.rankme_enabled = self.eval_freq > 0 and bool(rankme_cfg.get("enabled", False))
        self.rankme_every = max(1, int(rankme_cfg.get("every_events", 1)))
        self.subset_frac = float(rankme_cfg.get("subset_frac", 0.01))
        self.max_samples = int(rankme_cfg.get("max_samples", 4096))
        self.feature_mode = str(rankme_cfg.get("feature", "global")).lower()
        self.use_global = self.feature_mode in ("global", "both")
        self.use_tokens = self.feature_mode in ("tokens", "both")
        self.token_max_clips = int(rankme_cfg.get("token_max_clips", 128))
        self.rankme_seed = int(rankme_cfg.get("seed", self.seed))

        # -- Hessian
        self.hessian_enabled = self.eval_freq > 0 and bool(hessian_cfg.get("enabled", False))
        self.hessian_every = max(1, int(hessian_cfg.get("every_events", 1)))
        self.n_hutchinson = int(hessian_cfg.get("n_hutchinson", 5))
        self.n_clips = int(hessian_cfg.get("n_clips", 1))
        self.hessian_seed = int(hessian_cfg.get("seed", self.seed))
        self.loss_exp = float(loss_exp)

        # Fixed, deterministic global subset of sample indices.
        self._used_indices: list = []
        self._eval_ds = None  # deterministic clone of the training dataset
        self._rankme_loader = None  # lazily built, reused across events
        self._hessian_sample = None  # lazily cached on rank 0 (clips + masks)
        self._mask_collator = None  # private MaskCollator for Hessian masks

    # ------------------------------------------------------------------ #
    # public helpers
    # ------------------------------------------------------------------ #
    @property
    def any_enabled(self) -> bool:
        return self.rankme_enabled or self.hessian_enabled

    def ensure_built(self):
        """Prepare deterministic subset/dataset once. Cheap; call on all ranks."""
        if self._used_indices:
            return
        total = len(self.train_dataset)
        if total <= 0:
            return
        target = int(total * self.subset_frac)
        if self.rankme_enabled:
            target = min(target, self.max_samples)
        # small but at least the Hessian clips
        target = max(target, min(self.n_clips, total))
        target = min(target, total)
        rng = random.Random(self.rankme_seed)
        self._used_indices = sorted(rng.sample(range(total), target))
        if not self._used_indices:
            return
        transform = build_center_crop_transform(self.crop_size)
        self._eval_ds = _copy_dataset_deterministic(self.train_dataset, transform)

    def rankme_can_run(self) -> bool:
        self.ensure_built()
        return self.rankme_enabled and len(self._used_indices) > 0

    # ------------------------------------------------------------------ #
    # RankMe
    # ------------------------------------------------------------------ #
    def run_rankme(self, autocast_dtype=None) -> T.Dict[str, float]:
        """Extract encoder features over the fixed subset and return RankMe.

        Called on EVERY rank (each consumes its own slice, then all_gather).
        """
        metrics = {}
        self.ensure_built()
        if not self.rankme_can_run():
            return metrics
        used = self._used_indices
        per_rank = len(used) // self.world_size
        if per_rank <= 0:
            return metrics
        local_global_idx = used[self.rank * per_rank : (self.rank + 1) * per_rank]

        loader = self._make_rankme_loader(local_global_idx)
        context = torch.autocast(device_type="cuda", dtype=autocast_dtype) if (
            autocast_dtype is not None and self.device.type == "cuda"
        ) else contextlib.nullcontext()

        global_feats, token_vals, token_count = [], [], 0
        with _eval_mode(self.encoder):
            with torch.no_grad(), context:
                for batch in loader:
                    clip = batch[0].to(self.device, non_blocking=True)  # [B, C, T, H, W]
                    # encoder wrapper expects a per-dataset list of clips
                    tok = self.encoder([clip])
                    tok = tok[0].float()  # [B, n_tok, D]
                    if self.use_global:
                        global_feats.append(tok.mean(dim=1))
                    if self.use_tokens and token_count < self.token_max_clips:
                        for k in range(tok.shape[0]):
                            if token_count >= self.token_max_clips:
                                break
                            token_vals.append(_per_clip_token_rankme(tok[k]))
                            token_count += 1
        del tok

        if self.use_global:
            Z_local = torch.cat(global_feats, dim=0)  # fp32 on this device
            Z = _all_gather_rows(Z_local)
            del Z_local
            # column mean-center then RankMe
            Zc = Z.double().cpu()
            Zc = Zc - Zc.mean(dim=0)
            metrics.update(rankme_metrics(Zc))
            metrics["rankme_samples"] = int(Z.shape[0])
        if self.use_tokens and token_vals:
            # Per-clip RankMe values are summed across ranks (each rank scored a
            # disjoint slice); the mean weights every scored clip equally.
            total = torch.tensor([float(sum(token_vals)), float(token_count)], dtype=torch.float64)
            rank, world = _dist()
            if world > 1:
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            n_scored = int(total[1])
            metrics["rankme_tokens"] = float(total[0] / n_scored) if n_scored else float("nan")
        return metrics

    def _make_rankme_loader(self, local_global_idx):
        if self._rankme_loader is not None:
            return self._rankme_loader
        subset = Subset(self._eval_ds, local_global_idx)
        # `collate_fn`/`worker_init_fn` MUST be module-level (picklable) because
        # the DataLoader spawns worker processes (num_workers > 0) under torchrun.
        loader = DataLoader(
            subset,
            batch_size=8,
            shuffle=False,
            drop_last=False,
            num_workers=min(4, max(1, (os.cpu_count() or 4))),
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=_rankme_worker_init,
            collate_fn=_rankme_collate,
        )
        self._rankme_loader = loader
        return loader

    # ------------------------------------------------------------------ #
    # Hessian trace
    # ------------------------------------------------------------------ #
    def run_hessian(self) -> T.Dict[str, float]:
        """Hutchinson trace of the pretraining loss Hessian on a fixed cached batch.

        Heavy; performed only on rank 0, others join a barrier.
        """
        if not self.hessian_enabled:
            return {}
        self.ensure_built()
        if len(self._used_indices) == 0:
            return {}
        if self.rank == 0:
            metrics = self._hessian_on_rank0()
        else:
            metrics = None
        if self.world_size > 1:
            dist.barrier()
        return metrics if metrics is not None else {}

    def _hessian_on_rank0(self) -> T.Dict[str, float]:
        if self._hessian_sample is None:
            self._hessian_sample = self._build_hessian_sample()
        if self._hessian_sample is None:
            return {}
        clips, masks_enc, masks_pred = self._hessian_sample
        loss_exp = self.loss_exp
        params = [
            p
            for p in list(self.encoder.parameters()) + list(self.predictor.parameters())
            if p.requires_grad
        ]
        if not params:
            return {}
        try:
            torch.cuda.empty_cache()
            # _force_math_sdpa: flash / mem-efficient attention lack double-backward.
            with _eval_mode(self.encoder, self.predictor, self.target_encoder), \
                 _activation_checkpointing_disabled(self.encoder, self.predictor), \
                 _force_math_sdpa():
                clips = [c.to(self.device, non_blocking=True) for c in clips]
                masks_enc = [[m.to(self.device) for m in gen] for gen in masks_enc]
                masks_pred = [[m.to(self.device) for m in gen] for gen in masks_pred]
                loss = _pretrain_loss_on_sample(
                    self.encoder, self.predictor, self.target_encoder,
                    clips, masks_enc, masks_pred, loss_exp,
                )
                trace, trace_std = hutchinson_trace(loss, params, self.n_hutchinson, self.hessian_seed)
                del loss, clips
            torch.cuda.empty_cache()
            return {"hessian_trace": trace, "hessian_std": trace_std, "hessian_n": self.n_hutchinson}
        except Exception as e:  # surface but never let it kill training
            import logging

            logging.getLogger(__name__).warning(f"Hessian evaluation failed: {e}")
            return {}

    def _build_hessian_sample(self):
        """Fetch a tiny fixed batch of clips and generate masks once (rank 0)."""
        n_need = min(self.n_clips, len(self._used_indices))
        if n_need <= 0:
            return None
        if self._mask_collator is None:
            self._mask_collator = MaskCollator(
                cfgs_mask=self.cfgs_mask,
                dataset_fpcs=self.dataset_fpcs,
                crop_size=self.crop_size,
                patch_size=self.patch_size,
                tubelet_size=self.tubelet_size,
            )
        raw = []
        for gi in self._used_indices[: n_need * 4]:
            try:
                raw.append(self._eval_ds[gi])
            except Exception:
                continue
            if len(raw) >= n_need:
                break
        if len(raw) < 1:
            return None
        sample = self._mask_collator(raw)  # list of (udata, masks_enc, masks_pred) per fpc
        clips, masks_enc, masks_pred = [], [], []
        for fpc_sample in sample:
            udata, m_enc, m_pred = fpc_sample
            clips.append(udata[0][0].detach().cpu())
            masks_enc.append([m.detach().cpu() for m in m_enc])
            masks_pred.append([m.detach().cpu() for m in m_pred])
        return clips, masks_enc, masks_pred
