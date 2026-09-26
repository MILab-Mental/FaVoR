import math
import random
import torch


class AVTimeSampler:
    def __init__(self, global_seconds=6, local_seconds=2, local_views=4, training=True, short_policy="zero_pad", minimum_seconds=0.1):
        self.global_seconds, self.local_seconds = float(global_seconds), float(local_seconds)
        self.local_views, self.training = int(local_views), training
        self.short_policy = short_policy
        self.minimum_seconds = float(minimum_seconds)
        if short_policy not in {'error', 'zero_pad'}:
            raise ValueError('short_policy must be error or zero_pad')
        if not math.isfinite(self.minimum_seconds) or self.minimum_seconds <= 0:
            raise ValueError('minimum_seconds must be finite and positive')
        if self.global_seconds <= 0 or self.local_seconds <= 0 or self.local_seconds > self.global_seconds or self.local_views < 0:
            raise ValueError('invalid paired view durations/count')

    def __call__(self, video_duration, audio_duration):
        if not all(math.isfinite(v) for v in (video_duration, audio_duration)):
            raise ValueError('nonfinite media duration')
        available = min(video_duration, audio_duration)
        if available < self.minimum_seconds:
            raise ValueError('paired media is shorter than minimum_seconds')
        if available < self.global_seconds and self.short_policy == 'error':
            raise ValueError('paired media is shorter than global interval')
        duration = min(available, self.global_seconds)
        local_duration = min(duration, self.local_seconds)
        maximum = available - duration
        start = random.uniform(0, maximum) if self.training else maximum / 2
        global_interval = (start, start + duration)
        locals_ = []
        for k in range(self.local_views):
            maximum_offset = duration - local_duration
            offset = random.uniform(0, maximum_offset) if self.training else maximum_offset * (k + 1) / (self.local_views + 1)
            locals_.append((start + offset, start + offset + local_duration))
        return global_interval, locals_


def frame_indices(start, end, frames, real_fps, total_frames):
    if frames < 1 or real_fps <= 0:
        raise ValueError('invalid video frame count/FPS')
    targets = start + torch.arange(frames, dtype=torch.float64) * ((end - start) / frames)
    # Ceil ensures frame timestamps remain inside [start,end).
    indices = torch.ceil(targets * real_fps - 1e-9).long()
    times = indices.double() / real_fps
    if (indices >= total_frames).any() or (times >= end).any() or indices.unique().numel() != frames:
        raise ValueError('video interval would require out-of-range or repeated frames')
    return indices, times
