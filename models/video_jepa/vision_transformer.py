# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from .utils.masking import apply_masks
from .utils.modules import Block
from .utils.patch_embed import PatchEmbed, PatchEmbed3D
from .utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from utils.tensors import trunc_normal_


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        handle_nonsquare_inputs=True,
        use_cls_token=False,
        token_drop_rate=0.0,
        attn_mode="full",
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.use_cls_token = bool(use_cls_token)
        if not 0.0 <= float(token_drop_rate) < 1.0:
            raise ValueError(f"token_drop_rate must be in [0, 1), got {token_drop_rate}")
        self.token_drop_rate = float(token_drop_rate)
        if attn_mode not in ("full", "block_causal"):
            raise ValueError(f"attn_mode must be 'full' or 'block_causal', got {attn_mode!r}")
        self.attn_mode = attn_mode

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Position embedding
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + int(self.use_cls_token), embed_dim),
                requires_grad=False,
            )

        # Attention Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    num_prefix_tokens=int(self.use_cls_token),
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        if self.use_cls_token:
            trunc_normal_(self.cls_token, std=self.init_std)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=self.use_cls_token, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=self.use_cls_token)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None, return_tokens=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        # Image
        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        # Video
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        pos_embed = None
        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed[:, int(self.use_cls_token) :]

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # LeJEPA's random encoder-side token drop is independent per sample
        # and is active only during training.  Evaluation always sees the
        # complete dense patch grid.
        token_ids = masks
        if masks is None and self.training and self.token_drop_rate > 0:
            batch, num_patches, channels = x.shape
            keep = max(1, int(round(num_patches * (1.0 - self.token_drop_rate))))
            token_ids = torch.rand(batch, num_patches, device=x.device).argsort(dim=1)[:, :keep]
            x = torch.gather(x, 1, token_ids.unsqueeze(-1).expand(-1, -1, channels))

        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            if pos_embed is not None:
                cls = cls + pos_embed[:, :1]
            x = torch.cat([cls, x], dim=1)

        attn_mask = None
        if self.attn_mode == "block_causal":
            attn_mask = build_block_causal_mask(
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                token_ids=token_ids,
                num_prefix_tokens=int(self.use_cls_token),
                device=x.device,
            )

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, token_ids, attn_mask, T=T, H_patches=H_patches, W_patches=W_patches,
                    use_reentrant=False
                )
            else:
                x = blk(x, mask=token_ids, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        if return_tokens is False and self.use_cls_token:
            return x[:, 0]
        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N_total, dim = pos_embed.shape
        prefix = int(self.use_cls_token)
        cls_pos_embed = pos_embed[:, :prefix]
        pos_embed = pos_embed[:, prefix:]
        N = pos_embed.shape[1]

        if self.is_video:

            # If pos_embed already correct size, just return
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return torch.cat([cls_pos_embed, pos_embed], dim=1)

            # Just chop off last N tokens of positional embedding
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int((T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size))
                return torch.cat([cls_pos_embed, pos_embed[:, :new_N, :]], dim=1)

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return torch.cat([cls_pos_embed, pos_embed], dim=1)

        else:

            # If pos_embed already correct size, just return
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return torch.cat([cls_pos_embed, pos_embed], dim=1)

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat([cls_pos_embed, pos_embed], dim=1)


def build_block_causal_mask(
    T, H_patches, W_patches, token_ids=None, num_prefix_tokens=1, device=None
):
    """Build LeVJEPA block-causal attention (True means attention allowed).

    Patches attend bidirectionally within a frame and causally across frames.
    CLS is a readout register: it attends to the entire clip, while patches do
    not attend to CLS, preventing future information leaking backward through
    the prefix token in deeper layers.
    """
    tokens_per_frame = int(H_patches * W_patches)
    if token_ids is None:
        ids = torch.arange(int(T * tokens_per_frame), device=device).unsqueeze(0)
    else:
        ids = token_ids
    frame_ids = ids // tokens_per_frame
    mask = frame_ids.unsqueeze(-1) >= frame_ids.unsqueeze(-2)
    if num_prefix_tokens:
        batch, patches, _ = mask.shape
        prefix = int(num_prefix_tokens)
        full = mask.new_zeros((batch, patches + prefix, patches + prefix))
        full[:, :prefix, :] = True
        full[:, prefix:, prefix:] = mask
        mask = full
    return mask.unsqueeze(1)


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# We do not use any of the following ViT definitions in V-JEPA 2, but retain them for
# compatibility reasons.
def vit_synthetic(patch_size=16, **kwargs):
    # For performance testing only
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1,
        depth=1,
        num_heads=1,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant_xformers_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_gigantic(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=16,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_gigantic_xformers(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


VIT_EMBED_DIMS = {
    "vit_synthetic": 1,
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
    "vit_gigantic": 1664,
}
