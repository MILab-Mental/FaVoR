import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        mode="layer_norm",
        conv_bias=False,
        dropout=0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=conv_bias,
        )
        self.dropout = nn.Dropout(dropout)

        self.mode = mode
        if mode == "layer_norm":
            self.layer_norm = nn.LayerNorm(out_channels)
        elif mode == "group_norm":
            self.group_norm = nn.GroupNorm(out_channels, out_channels)
        else:
            self.layer_norm = None
            self.group_norm = None

        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: [B, C, T]
        return: [B, C_out, T_out]
        """
        x = self.conv(x)
        x = self.dropout(x)

        if self.mode == "layer_norm":
            # [B, C, T] -> [B, T, C]
            x = x.transpose(1, 2)
            x = self.layer_norm(x)
            # [B, T, C] -> [B, C, T]
            x = x.transpose(1, 2)
        elif self.mode == "group_norm":
            x = self.group_norm(x)

        x = self.activation(x)
        return x


class ConvFeatureExtractor(nn.Module):
    """
    emotion2vec / wav2vec2.0 style CNN feature extractor.

    Default 50Hz config:
      conv_layers = [
        [512, 10, 5],
        [512, 3, 2],
        [512, 3, 2],
        [512, 3, 2],
        [512, 3, 2],
        [512, 2, 2],
        [512, 2, 2],
      ]

    Total downsample:
      5 * 2 * 2 * 2 * 2 * 2 * 2 = 320

    16kHz / 320 = 50Hz.
    """

    def __init__(
        self,
        conv_layers,
        mode="layer_norm",
        conv_bias=False,
        dropout=0.0,
    ):
        super().__init__()
        blocks = []
        in_channels = 1

        for out_channels, kernel_size, stride in conv_layers:
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    mode=mode,
                    conv_bias=conv_bias,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.conv_blocks = nn.ModuleList(blocks)
        self.embedding_dim = in_channels

    def forward(self, x):
        """
        x: [B, T] or [B, 1, T]
        return: [B, N, C]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        for block in self.conv_blocks:
            x = block(x)

        # [B, C, N] -> [B, N, C]
        x = x.transpose(1, 2)
        return x

    def output_length(self, L: int) -> int:
        """波形采样点数 -> CNN 输出 token 数（与 forward 的 floor 语义一致）"""
        for block in self.conv_blocks:
            k = block.conv.kernel_size[0]
            s = block.conv.stride[0]
            L = (L - k) // s + 1
        return L
