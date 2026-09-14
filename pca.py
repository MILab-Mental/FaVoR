# Patch-token PCA RGB visualization for video.
# The full video is processed with a sliding window (default 48 frames / stride 24).
# Patch tokens [N, D] -> PCA -> top 3 components -> RGB channels (PC1->R, PC2->G,
# PC3->B), min-max normalize each channel to [0,1], then upsample the
# [H/p, W/p, 3] map back to the original video resolution.

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from sklearn.decomposition import PCA

import models.vision_transformer as video_vit
from utils.checkpoint_loader import robust_checkpoint_loader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-token PCA RGB visualization")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/data/sdc/FAVOR/DATASET/splits-0901/demo_video/2_RAVDESS_SV/v01.mp4",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/data/sdc/FAVOR/OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/latest.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/data/sdc/FAVOR/OUTPUT/pretrain_v/vitl16/FaVoR-112px-48f/2_RAVDESS_SV_v01.mp4",
    )
    parser.add_argument("--pca-mode", type=str, default="clip", choices=["clip", "frame"])
    parser.add_argument("--interp", type=str, default="nearest", choices=["nearest", "bilinear"])
    parser.add_argument("--num-frames", type=int, default=48,
                        help="context window size in frames per forward pass")
    parser.add_argument("--window-stride", type=int, default=24,
                        help="stride between windows; 24 = ~1 PCA frame per input frame")
    parser.add_argument("--crop-size", type=int, default=1344,
                        help="model input resolution; patch grid = crop_size/patch_size (1344 -> 112x112)")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    return parser.parse_args()


def load_encoder(args, device):
    encoder = video_vit.__dict__["vit_large"](
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
    )
    ckpt = robust_checkpoint_loader(args.ckpt, map_location="cpu")
    state = ckpt["encoder"]
    model_state = encoder.state_dict()
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k in model_state and model_state[k].shape == v.shape:
            cleaned[k] = v
    msg = encoder.load_state_dict(cleaned, strict=False)
    print(f"loaded encoder from {args.ckpt}: {msg}")
    encoder.to(device)
    encoder.eval()
    return encoder


def window_offsets(total, window, stride):
    """Start indices of sliding windows covering the whole video."""
    if total <= window:
        return [0]
    offs = list(range(0, total - window + 1, stride))
    if offs[-1] != total - window:
        offs.append(total - window)
    return offs


def preprocess_window(vr, start, end, args):
    idx = list(range(start, end))
    frames = vr.get_batch(idx).asnumpy()  # [T, H, W, 3] uint8
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    clip = F.interpolate(clip, size=(args.crop_size, args.crop_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (clip - mean) / std


def pca_all(all_tokens, k, t, hp, wp, mode, max_fit=100000, seed=0):
    """all_tokens: [K*t*hp*wp, D] float32 -> [K, t, hp, wp, 3] in [0,1]."""
    if mode == "clip":
        # Single PCA over all tokens of the whole video -> temporally stable colors.
        x = all_tokens
        n = x.shape[0]
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(n, max_fit, replace=False) if n > max_fit else np.arange(n)
        pca = PCA(n_components=3).fit(x[fit_idx])
        comps = pca.transform(x).reshape(k, t, hp, wp, 3)
        lo = comps.min(axis=(0, 1, 2, 3), keepdims=True)
        hi = comps.max(axis=(0, 1, 2, 3), keepdims=True)
        comps = (comps - lo) / (hi - lo + 1e-8)
    else:
        # Independent PCA per frame (may flicker due to component sign flips).
        feats = all_tokens.reshape(k * t, hp * wp, -1)
        comps = np.zeros((k * t, hp, wp, 3), dtype=np.float32)
        for i in range(k * t):
            p = PCA(n_components=3).fit_transform(feats[i].astype(np.float64))
            p = (p - p.min(axis=0)) / (p.max(axis=0) - p.min(axis=0) + 1e-8)
            comps[i] = p.reshape(hp, wp, 3)
        comps = comps.reshape(k, t, hp, wp, 3)
    return comps


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    encoder = load_encoder(args, device)

    vr = VideoReader(args.input, num_threads=1, ctx=cpu(0))
    total = len(vr)
    fps = float(vr.get_avg_fps())
    orig_h, orig_w = vr[0].asnumpy().shape[:2]

    hp = args.crop_size // args.patch_size
    wp = args.crop_size // args.patch_size
    t = args.num_frames // args.tubelet_size
    offs = window_offsets(total, args.num_frames, args.window_stride)
    print(f"input: {args.input} ({total} frames, {fps:.2f} fps, {orig_w}x{orig_h})")
    print(f"{len(offs)} windows of {args.num_frames} frames, stride {args.window_stride}")

    use_amp = device == "cuda"
    feats = []
    for wi, start in enumerate(offs):
        end = min(start + args.num_frames, total)
        clip = preprocess_window(vr, start, end, args)
        # [T, 3, H, W] -> [1, 3, T, H, W] (PatchEmbed3D expects channels-first)
        x = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = encoder(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        feats.append(out[0].float().cpu().numpy().astype(np.float32))
        print(f"window {wi + 1}/{len(offs)}: frames [{start}, {end}) -> tokens {feats[-1].shape}")

    all_tokens = np.concatenate(feats, axis=0)
    print(f"all tokens: {all_tokens.shape}")
    comps = pca_all(all_tokens, len(offs), t, hp, wp, args.pca_mode)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {args.output}")

    total_out = len(offs) * t
    written = 0
    for k in range(len(offs)):
        for i in range(t):
            rgb = torch.from_numpy(comps[k, i]).permute(2, 0, 1).unsqueeze(0).float()
            rgb = F.interpolate(rgb, size=(orig_h, orig_w), mode=args.interp)
            frame = rgb[0].permute(1, 2, 0).numpy()
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            written += 1
            if written % 50 == 0 or written == total_out:
                print(f"wrote {written}/{total_out} frames")
    writer.release()
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
