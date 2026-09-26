from dataclasses import dataclass
from torch import Tensor


@dataclass
class TemporalTokenOutput:
    tokens: Tensor
    token_times: Tensor
    padding_mask: Tensor
    pooled: Tensor
    temporal_size: int
    spatial_size: int | None = None


@dataclass
class AlignedTokens:
    video: TemporalTokenOutput
    audio: TemporalTokenOutput
    bin_edges: Tensor
    assignment: Tensor
    audio_features: Tensor
    audio_padding_mask: Tensor
    audio_counts: Tensor
