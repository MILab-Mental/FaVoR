import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedQKVSelfAttention(nn.Module):
    """
    Compatible with emotion2vec_plus_large:
      blocks.i.attn.qkv.weight
      blocks.i.attn.qkv.bias
      blocks.i.attn.proj.weight
      blocks.i.attn.proj.bias
    """

    def __init__(self, d_model, nhead, dropout=0.0, qkv_bias=True):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, N, D]
        key_padding_mask: [B, N], True means ignored.
        """
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class PreNormMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PreNormTransformerBlock(nn.Module):
    """
    Compatible with emotion2vec_plus_large blocks:

      blocks.i.norm1
      blocks.i.attn.qkv
      blocks.i.attn.proj
      blocks.i.norm2
      blocks.i.mlp.fc1
      blocks.i.mlp.fc2
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        qkv_bias=True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FusedQKVSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = PreNormMLP(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PreNormTransformerEncoder(nn.Module):
    def __init__(
        self,
        depth,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        qkv_bias=True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            PreNormTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None, return_all_layers=False):
        all_layers = []

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            if return_all_layers:
                all_layers.append(x)

        if return_all_layers:
            return x, all_layers

        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, N, D]
        key_padding_mask: [B, N], True means ignored / masked.
        """
        B, N, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            # key_padding_mask: [B, N] True -> ignore
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """
    Post-norm Transformer block, fairseq/data2vec style.

    Layer names are designed to be compatible with common fairseq checkpoints:
      self_attn.q_proj / k_proj / v_proj / out_proj
      fc1 / fc2
      self_attn_layer_norm
      final_layer_norm
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
    ):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.activation = nn.GELU()

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, N, D]
        key_padding_mask: [B, N], True means ignored.
        """
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.dropout1(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        use_final_norm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.layer_norm = nn.LayerNorm(d_model) if use_final_norm else None

    def forward(self, x, key_padding_mask=None, return_all_layers=False):
        """
        x: [B, N, D]
        key_padding_mask: [B, N], True means ignored.
        """
        all_layers = []

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            if return_all_layers:
                all_layers.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if return_all_layers:
            return x, all_layers

        return x
