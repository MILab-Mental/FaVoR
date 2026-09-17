import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models.attentive_pooler import AttentiveClassifier
from models import vision_transformer as vit

LOGGER = logging.getLogger(__name__)


class ClipEncoder(nn.Module):
    """V-JEPA multilevel multi-clip token aggregation for fine-tuning.

    The backbone is configured with ``out_layers`` and returns one token tensor
    per selected transformer block. Those tensors are concatenated along the
    token dimension before clips are unrolled and joined along time, matching
    V-JEPA's ``vit_encoder_multiclip_multilevel`` evaluation wrapper.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    def forward(self, clips):
        # DataLoader returns [num_clips][num_views][B,C,T,H,W].
        num_clips = len(clips)
        num_views = len(clips[0])
        batch, _, frames, _, _ = clips[0][0].shape

        inputs = torch.cat([torch.cat(clip, dim=0) for clip in clips], dim=0)
        layer_outputs = self.backbone(inputs)
        if not isinstance(layer_outputs, (list, tuple)):
            raise RuntimeError("ClipEncoder requires backbone out_layers to return multilevel token tensors")
        outputs = torch.cat(layer_outputs, dim=1)
        _, token_count, embed_dim = outputs.shape
        temporal_tokens = frames // self.backbone.tubelet_size
        if token_count % temporal_tokens:
            raise RuntimeError(
                f"Cannot reshape {token_count} tokens into {temporal_tokens} temporal positions; "
                "check model.out_layers and input frame count."
            )
        spatial_tokens = token_count // temporal_tokens

        effective_batch = batch * num_views
        all_views = [[] for _ in range(num_views)]
        for clip_index in range(num_clips):
            clip_outputs = outputs[clip_index * effective_batch : (clip_index + 1) * effective_batch]
            for view_index in range(num_views):
                all_views[view_index].append(
                    clip_outputs[view_index * batch : (view_index + 1) * batch]
                )

        return [
            torch.cat([output.reshape(batch, temporal_tokens, spatial_tokens, embed_dim) for output in view_outputs], dim=1)
            .flatten(1, 2)
            for view_outputs in all_views
        ]


class AttentiveRegressor(AttentiveClassifier):
    """Attentive pooling head for a scalar regression target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_classes=1, **kwargs)


def _encoder_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("encoder", "target_encoder", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_backbone(backbone, checkpoint_path):
    if not checkpoint_path:
        LOGGER.info("No pretrained checkpoint supplied; using random initialization")
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _encoder_state(checkpoint)
    cleaned = {}
    for key, value in source.items():
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        cleaned[key] = value
    current = backbone.state_dict()
    compatible = {k: v for k, v in cleaned.items() if k in current and current[k].shape == v.shape}
    msg = backbone.load_state_dict(compatible, strict=False)
    LOGGER.info("Loaded %d/%d compatible encoder tensors from %s; missing=%d unexpected=%d",
                len(compatible), len(current), checkpoint_path, len(msg.missing_keys), len(msg.unexpected_keys))
    return checkpoint


def build_model(
    checkpoint_path,
    *,
    model_name,
    crop_size,
    patch_size,
    max_num_frames,
    tubelet_size,
    task,
    num_class=None,
    out_layers=None,
    classifier_depth=1,
    classifier_num_heads=None,
    uniform_power=False,
    use_activation_checkpointing=False,
    use_rope=False,
    use_sdpa=False,
    use_silu=False,
    wide_silu=True,
):
    """Build a fine-tuning encoder and task head from configuration values.

    The checkpoint is used exclusively for the backbone.  Fine-tuning heads
    are always freshly initialized because their output dimension is specific
    to the downstream task.
    """
    if task not in {"classification", "regression", "multi_label_classification"}:
        raise ValueError(
            f"Unsupported task {task!r}; use 'classification', 'multi_label_classification', or 'regression'."
        )
    if task in {"classification", "multi_label_classification"} and (not isinstance(num_class, int) or num_class < 2):
        raise ValueError(f"{task} requires data.num_class to be an integer of at least 2.")
    if not isinstance(out_layers, (list, tuple)) or not out_layers:
        raise ValueError("model.out_layers must be a non-empty list of transformer block indices")

    try:
        backbone_factory = vit.__dict__[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown video encoder model_name: {model_name!r}") from error

    backbone = backbone_factory(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        wide_silu=wide_silu,
        out_layers=out_layers,
    )
    checkpoint = load_backbone(backbone, checkpoint_path)
    encoder = ClipEncoder(backbone)
    head_kwargs = dict(
        embed_dim=encoder.embed_dim,
        num_heads=classifier_num_heads or backbone.num_heads,
        depth=classifier_depth,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    if task in {"classification", "multi_label_classification"}:
        # Multi-label reuses the same logit head; sigmoid + BCE is applied in the loss.
        head = AttentiveClassifier(**head_kwargs, num_classes=num_class)
    else:
        head = AttentiveRegressor(**head_kwargs)

    return encoder, head
