import json
import math
import subprocess
import torch
from datasets.audio_jepa.pretrain_dataset import load_audio
from .av_time_sampler import frame_indices


def probe_media(path, kind):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0' if kind == 'video' else 'a:0',
        '-show_entries', 'stream=start_time,duration,sample_rate,avg_frame_rate,nb_frames:format=start_time,duration',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    if not payload.get('streams'):
        raise ValueError(f'{path}: no {kind} stream')
    stream, container = payload['streams'][0], payload.get('format', {})
    duration = stream.get('duration', container.get('duration'))
    start = stream.get('start_time', container.get('start_time', 0))
    fps = stream.get('avg_frame_rate', '0/1').split('/')
    frame_rate = float(fps[0]) / max(float(fps[1]), 1) if len(fps) == 2 else float(fps[0])
    return {'duration': float(duration), 'start': float(start), 'sample_rate': int(stream.get('sample_rate', 0)), 'fps': frame_rate}


def validate_sync(video, audio, row, cfg):
    if cfg.get('strict', True) is not True or cfg.get('on_mismatch', 'error') != 'error':
        raise ValueError('VA v1 requires strict sync with on_mismatch=error')
    for flag in ('require_same_source_id', 'require_same_start', 'require_same_duration'):
        if not cfg.get(flag, True):
            raise ValueError(f'VA strict sync requires {flag}')
    if not all(math.isfinite(media[key]) for media in (video, audio) for key in ('duration', 'start')):
        raise ValueError('invalid media timing metadata')
    duration_delta = abs(video['duration'] - audio['duration'])
    start_delta = abs(video['start'] - audio['start'])
    offset = float(row.get('av_offset_seconds') or 0)
    tolerance = float(cfg.get('duration_tolerance_seconds', .5))
    frame_tolerance = float(cfg.get('duration_tolerance_frames', 2))
    fps = float(video.get('fps', 0))
    if fps > 0:
        tolerance = max(tolerance, frame_tolerance / fps)
    offset_tolerance = float(cfg.get('offset_tolerance_seconds', .1))
    if any(not math.isfinite(v) or v < 0 for v in (tolerance, frame_tolerance, offset_tolerance)) or not math.isfinite(offset):
        raise ValueError('invalid synchronization tolerance/offset')
    audio_shift = audio['start'] - video['start'] + offset
    if duration_delta > tolerance or start_delta > offset_tolerance or abs(offset) > offset_tolerance or abs(audio_shift) > offset_tolerance:
        raise ValueError(f'pair {row["pair_id"]}: sync mismatch duration_delta={duration_delta:.6f} start_delta={start_delta:.6f} offset={offset}')
    return dict(duration_tolerance_seconds=tolerance, audio_shift_seconds=audio_shift, video_duration=video['duration'], audio_duration=audio['duration'], duration_delta=duration_delta,
                start_delta=start_delta, av_offset_seconds=offset, source_sample_rate=audio['sample_rate'], repeated=False, padded=False)


class AVDecoder:
    def __init__(self, row, sample_rate, sync_cfg):
        from decord import VideoReader, cpu
        self.reader = VideoReader(row['video_path'], ctx=cpu(0), num_threads=1)
        self.fps = float(self.reader.get_avg_fps())
        video_meta = probe_media(row['video_path'], 'video')
        video_meta['fps'] = self.fps
        audio_meta = probe_media(row['audio_path'], 'audio')
        self.diagnostics = validate_sync(video_meta, audio_meta, row, sync_cfg)
        self.diagnostics['video_fps'] = self.fps
        self.waveform = load_audio(row['audio_path'], sample_rate)
        self.sample_rate = sample_rate
        decoded_audio_duration = self.waveform.numel() / sample_rate
        if abs(decoded_audio_duration - audio_meta['duration']) > self.diagnostics['duration_tolerance_seconds']:
            raise ValueError('decoded audio length differs from probed duration')
        # Positive shift means audio begins later on the video's time axis.
        shift = self.diagnostics['audio_shift_seconds']
        self.video_origin = max(0., shift)
        self.audio_origin = self.video_origin - shift
        common_end = min(video_meta['duration'], len(self.reader) / self.fps, shift + decoded_audio_duration)
        self.video_duration = self.audio_duration = common_end - self.video_origin
        if self.video_duration <= 0:
            raise ValueError('paired media has no common time interval')
        self.diagnostics.update(common_duration=self.video_duration,
            cropped=abs(shift) > 0 or video_meta['duration'] != decoded_audio_duration)

    def decode(self, interval, frames):
        start, end = interval
        indices, times = frame_indices(start + self.video_origin, end + self.video_origin, frames, self.fps, len(self.reader))
        times = times - self.video_origin
        begin, finish = round((start + self.audio_origin) * self.sample_rate), round((end + self.audio_origin) * self.sample_rate)
        if begin < 0 or finish > self.waveform.numel() or finish <= begin:
            raise ValueError('audio crop out of bounds; strict mode forbids padding')
        video = torch.from_numpy(self.reader.get_batch(indices.tolist()).asnumpy())
        targets = start + torch.arange(frames, dtype=torch.float64) * ((end - start) / frames)
        return dict(video_sampling_drift_seconds=(times - targets).abs().max(),
            audio_rounding_delta_seconds=torch.tensor([begin / self.sample_rate - self.audio_origin - start, finish / self.sample_rate - self.audio_origin - end], dtype=torch.float64),
            video=video, audio=self.waveform[begin:finish].clone(), audio_lengths=torch.tensor(finish - begin),
            start_time=torch.tensor(start, dtype=torch.float64), end_time=torch.tensor(end, dtype=torch.float64),
            audio_time_origin=torch.tensor(self.audio_origin, dtype=torch.float64),
            video_frame_indices=indices, video_frame_times=times, audio_sample_range=torch.tensor([begin, finish]))
