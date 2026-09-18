import torch
import torch.nn as nn

from .positional import FixedPositionalEmbedding
from .transformer import TransformerBlock


class JEPAPredictor(nn.Module):
    """
    WavJEPA-style full-sequence predictor.

    Input:
      context_features_full: [B, N, encoder_dim]
          Full sequence output from context encoder.
      context_mask: [B, N]
          True means non-context / ignored.
      target_positions: [B, K, N]
          Bool tensor. True means target token for target group k.

    Output:
      preds: [B*K, N, encoder_dim]
          Full sequence predictions. Loss should be computed only on target positions.
    """

    def __init__(
        self,
        encoder_dim=1024,
        decoder_dim=384,
        depth=12,
        nhead=12,
        dim_feedforward=1536,
        total_patches=100,
        dropout=0.0,
    ):
        """
        WavJEPA-style full-sequence predictor.

        For emotion2vec_plus_large:
            encoder_dim should be 1024.

        For original WavJEPA / emotion2vec base style:
            encoder_dim may be 768.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.total_patches = total_patches

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = FixedPositionalEmbedding(total_patches, decoder_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=decoder_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)
        self.decoder_to_encoder = nn.Linear(decoder_dim, encoder_dim)

    def forward(self, context_features_full, context_mask, target_positions):
        B, N, _ = context_features_full.shape
        K = target_positions.size(1)

        # Map context encoder features to predictor dimension.
        ctx = self.encoder_to_decoder(context_features_full)  # [B, N, decoder_dim]

        # Start from full sequence mask tokens.
        # NOTE: mask_token 是 nn.Parameter（float32），.repeat() 不受 autocast 影响，
        # 而 ctx 在 bf16 autocast 下已经是 bfloat16，两者 dtype 不一致时
        # tgt[visible] = ctx[visible] 这种 index_put 会直接报错，必须显式对齐。
        tgt = self.mask_token.repeat(B, N, 1).to(dtype=ctx.dtype)  # [B, N, decoder_dim]

        # Fill visible context positions back into full sequence.
        visible = ~context_mask  # [B, N]
        tgt[visible] = ctx[visible]

        # Add fixed decoder positional embedding.
        tgt = self.pos_embed(tgt)

        # Repeat for each target group.
        tgt = tgt.unsqueeze(1).repeat(1, K, 1, 1)  # [B, K, N, D]
        tgt = tgt.view(B * K, N, self.decoder_dim)

        # Official WavJEPA-style combined mask:
        # context_mask: True = ignored non-context
        # target_positions: True = target tokens
        # logical_xor makes context and current target visible,
        # while other non-context non-target tokens are ignored.
        ignore = torch.logical_xor(
            context_mask.unsqueeze(1),   # [B, 1, N]
            target_positions             # [B, K, N]
        )
        ignore = ignore.view(B * K, N)

        x = tgt
        for block in self.blocks:
            x = block(x, key_padding_mask=ignore)

        x = self.norm(x)
        preds = self.decoder_to_encoder(x)  # [B*K, N, encoder_dim]

        return preds
