import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .projector import Projector
from .encoder import build_encoder

logger = logging.getLogger(__name__)


class VideoLeJEPA(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def encode(self, video):
        return self.encoder(video, return_tokens=False)

    def forward(self, global_video, local_video):
        if local_video.ndim != 6:
            raise ValueError(f"local_video must be [B,K,C,T,H,W], got {tuple(local_video.shape)}")
        batch, views = local_video.shape[:2]
        global_cls = self.encode(global_video).unsqueeze(1)
        local_cls = self.encode(local_video.flatten(0, 1)).reshape(batch, views, -1)
        return self.projector(torch.cat([global_cls, local_cls], dim=1))


def build_video_lejepa(model_cfg):
    projector_cfg = model_cfg.get("projector", {})
    encoder = build_encoder(
        model_name=model_cfg.get("name", model_cfg.get("model_name", "vit_large")),
        img_size=model_cfg.get("global_size", model_cfg.get("img_size", 224)),
        patch_size=model_cfg.get("patch_size", 16),
        num_frames=model_cfg.get("num_frames", 16),
        tubelet_size=model_cfg.get("tubelet_size", 1),
        uniform_power=model_cfg.get("uniform_power", False),
        use_rope=model_cfg.get("use_rope", True),
        use_sdpa=model_cfg.get("use_sdpa", True),
        use_silu=model_cfg.get("use_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", False),
        token_drop_rate=model_cfg.get("token_drop_rate", 0.95),
        attn_mode=model_cfg.get("attn_mode", "block_causal"),
    )
    projector = Projector(
        encoder.embed_dim,
        hidden_dim=projector_cfg.get("hidden_dim", 2048),
        output_dim=projector_cfg.get("output_dim", 256),
    )
    return VideoLeJEPA(encoder, projector)


def _extract_encoder_state(checkpoint):
    for key in ("encoder", "target_encoder", "state_dict", "model"):
        value = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(value, dict):
            return value
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint has no encoder/state_dict/model tensor mapping")


def _strip_encoder_prefix(key):
    prefixes = ("module.", "encoder.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_vjepa_encoder(encoder, checkpoint_path, min_load_ratio=0.90):
    # FAVOR checkpoints can include multi-gigabyte optimizer state.  mmap keeps
    # conversion bounded to the encoder tensors actually copied below.
    checkpoint = torch.load(
        str(Path(checkpoint_path)), map_location="cpu", weights_only=False, mmap=True
    )
    source = _extract_encoder_state(checkpoint)
    target = encoder.state_dict()
    compatible, unexpected, mismatched = {}, [], []
    for raw_key, value in source.items():
        key = _strip_encoder_prefix(raw_key)
        if key not in target:
            unexpected.append(raw_key)
        elif target[key].shape != value.shape:
            mismatched.append((raw_key, tuple(value.shape), tuple(target[key].shape)))
        else:
            compatible[key] = value

    # A tubelet-2 Conv3d checkpoint can initialize tubelet-1 without changing
    # the requested architecture: average the temporal kernel into one slice.
    conv_key = "patch_embed.proj.weight"
    for raw_key, source_shape, target_shape in list(mismatched):
        if _strip_encoder_prefix(raw_key) == conv_key:
            value = source[raw_key]
            if value.ndim == 5 and target[conv_key].ndim == 5 and value.shape[:2] == target[conv_key].shape[:2] and value.shape[3:] == target[conv_key].shape[3:] and target[conv_key].shape[2] == 1:
                compatible[conv_key] = value.mean(dim=2, keepdim=True)
                mismatched.remove((raw_key, source_shape, target_shape))
                logger.info("converted %s temporal kernel %s -> %s by averaging", raw_key, source_shape, target_shape)
                break

    # FAVOR's absolute positional embeddings contain patch positions only.
    # Interpolate them to the requested tubelet/grid shape when necessary and
    # prepend the new zero-valued CLS position used by LeJEPA.
    pos_key = "pos_embed"
    for raw_key, source_shape, target_shape in list(mismatched):
        if _strip_encoder_prefix(raw_key) != pos_key:
            continue
        value = source[raw_key]
        target_value = target[pos_key]
        if value.ndim == 3 and target_value.ndim == 3 and value.shape[0] == target_value.shape[0] and value.shape[2] == target_value.shape[2]:
            spatial = (encoder.img_height // encoder.patch_size) * (encoder.img_width // encoder.patch_size)
            source_tokens = value.shape[1]
            target_tokens = target_value.shape[1] - 1
            if source_tokens % spatial == 0 and target_tokens % spatial == 0:
                source_t = source_tokens // spatial
                target_t = target_tokens // spatial
                height = encoder.img_height // encoder.patch_size
                width = encoder.img_width // encoder.patch_size
                patches = value.reshape(1, source_t, height, width, value.shape[-1]).permute(0, 4, 1, 2, 3)
                patches = F.interpolate(patches, size=(target_t, height, width), mode="trilinear", align_corners=False)
                patches = patches.permute(0, 2, 3, 4, 1).reshape(1, target_tokens, value.shape[-1])
                compatible[pos_key] = torch.cat([torch.zeros_like(patches[:, :1]), patches], dim=1)
                mismatched.remove((raw_key, source_shape, target_shape))
                logger.info("converted %s positional embedding %s -> %s with CLS prefix", raw_key, source_shape, target_shape)
        break

    message = encoder.load_state_dict(compatible, strict=False)
    loaded_numel = sum(target[key].numel() for key in compatible)
    eligible_numel = sum(value.numel() for key, value in target.items() if key != "cls_token")
    ratio = loaded_numel / max(eligible_numel, 1)
    logger.info("V-JEPA encoder load: %d tensors, %.2f%% parameters", len(compatible), ratio * 100)
    if message.missing_keys:
        logger.warning("missing encoder keys: %s", message.missing_keys)
    if unexpected:
        logger.warning("unexpected/skipped checkpoint keys: %s", unexpected)
    if mismatched:
        logger.warning("shape-mismatched checkpoint keys: %s", mismatched)
    if ratio < float(min_load_ratio):
        raise RuntimeError(
            f"V-JEPA encoder load ratio {ratio:.4f} is below min_load_ratio={min_load_ratio}"
        )
    return {"loaded_ratio": ratio, "missing": message.missing_keys, "unexpected": unexpected, "mismatched": mismatched}
