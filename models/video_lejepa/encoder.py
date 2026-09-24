from models.video_jepa import vision_transformer as video_vit


class LeVideoEncoder(video_vit.VisionTransformer):
    """Opt-in LeJEPA specialization of FAVOR's existing video ViT backbone."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_cls_token", True)
        kwargs.setdefault("token_drop_rate", 0.95)
        kwargs.setdefault("attn_mode", "block_causal")
        kwargs.setdefault("tubelet_size", 1)
        super().__init__(*args, **kwargs)


def build_encoder(model_name="vit_large", **kwargs):
    factory = video_vit.__dict__.get(model_name)
    if factory is None:
        raise ValueError(f"unknown video backbone {model_name!r}")
    # Factories construct the shared VisionTransformer class with exactly the
    # same parameter tree.  The opt-in switches provide LeVideoEncoder semantics.
    return factory(use_cls_token=True, **kwargs)
