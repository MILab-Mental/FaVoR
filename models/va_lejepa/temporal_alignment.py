"""Physical-time alignment; tensor lengths never determine the time mapping."""
import torch
from .temporal_types import AlignedTokens


def cnn_geometry(extractor):
    jump, receptive_field, first_center = 1, 1, 0.0
    for block in extractor.conv_blocks:
        conv = block.conv
        kernel, stride = conv.kernel_size[0], conv.stride[0]
        dilation, padding = conv.dilation[0], conv.padding[0]
        first_center += ((kernel - 1) * dilation / 2 - padding) * jump
        receptive_field += (kernel - 1) * dilation * jump
        jump *= stride
    return jump, receptive_field, first_center


def audio_token_times(extractor, count, sample_rate, sample_start, device):
    jump, _, first_center = cnn_geometry(extractor)
    centers = (first_center + torch.arange(count, device=device, dtype=torch.float64) * jump) / sample_rate
    return sample_start.to(device=device, dtype=torch.float64)[:, None] + centers[None]


def video_bin_edges(frame_times, starts, ends):
    if frame_times.ndim != 2 or frame_times.shape[1] < 1:
        raise ValueError('frame_times must be [B,T]')
    if not torch.isfinite(frame_times).all() or (frame_times[:, 1:] <= frame_times[:, :-1]).any():
        raise ValueError('video frame timestamps must be finite and strictly increasing')
    if (frame_times[:, 0] < starts - 1e-8).any() or (frame_times[:, -1] >= ends).any():
        raise ValueError('video frame timestamps are outside paired interval')
    return torch.cat((starts[:, None], (frame_times[:, 1:] + frame_times[:, :-1]) / 2, ends[:, None]), 1)


def align_audio_to_video(video, audio, starts, ends):
    edges = video_bin_edges(video.token_times, starts, ends)
    # right=True implements [start,end), including exact internal boundaries.
    assignment = torch.searchsorted(edges.contiguous(), audio.token_times.contiguous(), right=True) - 1
    valid = (~audio.padding_mask) & (assignment >= 0) & (assignment < video.temporal_size)
    safe_ids = assignment.clamp(0, video.temporal_size - 1)
    counts = torch.zeros(video.tokens.shape[0], video.temporal_size, device=video.tokens.device, dtype=torch.long)
    counts.scatter_add_(1, safe_ids, valid.long())
    features = audio.tokens.new_zeros(video.tokens.shape[0], video.temporal_size, audio.tokens.shape[-1])
    features.scatter_add_(1, safe_ids[..., None].expand_as(audio.tokens), audio.tokens.masked_fill(~valid[..., None], 0))
    features = features / counts.clamp_min(1)[..., None]
    assignment = assignment.masked_fill(~valid, -1)
    return AlignedTokens(video, audio, edges, assignment, features, counts == 0, counts)
