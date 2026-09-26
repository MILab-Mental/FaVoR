import torch
from .temporal_types import TemporalTokenOutput
from .temporal_alignment import audio_token_times


def video_token_output(encoder, view):
    if encoder.tubelet_size != 1 or not encoder.use_cls_token or encoder.token_drop_rate != 0:
        raise ValueError('VA dense path requires CLS, tubelet_size=1, token_drop_rate=0; sparse tokens need token_ids')
    clips = view['video']
    _, _, temporal, height, width = clips.shape
    spatial = (height // encoder.patch_size) * (width // encoder.patch_size)
    tokens = encoder(clips, return_tokens=True)
    if tokens.shape[1] != 1 + temporal * spatial:
        raise ValueError('dense video token count does not match exact spatiotemporal grid')
    times = view['video_frame_times'].to(device=tokens.device, dtype=torch.float64) - view['start_time'].to(tokens.device)[:, None]
    if times.shape != (clips.shape[0], temporal):
        raise ValueError('video timestamps do not match frame count')
    return TemporalTokenOutput(tokens[:, 1:].reshape(clips.shape[0], temporal, spatial, -1), times,
                               torch.zeros_like(times, dtype=torch.bool), tokens[:, 0], temporal, spatial)


def audio_token_output(encoder, view, sample_rate, checkpointing=False):
    if checkpointing and encoder.training and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        def encode(waves, lengths):
            output = encoder(waves, lengths, return_tokens=True)
            return output.embedding, output.tokens, output.padding_mask
        pooled, tokens, mask = checkpoint(encode, view['audio'], view['audio_lengths'], use_reentrant=False)
    else:
        output = encoder(view['audio'], view['audio_lengths'], return_tokens=True)
        pooled, tokens, mask = output.embedding, output.tokens, output.padding_mask
    times = audio_token_times(encoder.backbone.feature_extractor, tokens.shape[1], sample_rate,
                              view['audio_sample_range'][:, 0].double() / sample_rate, tokens.device)
    if 'audio_time_origin' in view:
        times = times - view['audio_time_origin'].to(tokens.device)[:, None]
    times = times - view['start_time'].to(tokens.device)[:, None]
    return TemporalTokenOutput(tokens, times, mask, pooled, tokens.shape[1])
