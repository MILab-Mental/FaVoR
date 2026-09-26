import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .va_manifest import read_manifest
from .av_time_sampler import AVTimeSampler
from .av_decoder import AVDecoder
from .transforms import AVTransforms
from .collate import collate_va_lejepa


class VALeJEPADataset(Dataset):
    def __init__(self, cfg, training=True, decoder_factory=AVDecoder):
        self.cfg = dict(cfg)
        self.training = training
        self.epoch = 0
        self.on_decode_error = cfg.get('on_decode_error', 'error')
        if self.on_decode_error not in {'error', 'skip'}:
            raise ValueError('on_decode_error must be error or skip')
        manifests = cfg.get('manifests', cfg.get('datasets', []))
        if isinstance(manifests, str):
            manifests = [manifests]
        self.rows = []
        for path in manifests:
            self.rows.extend(read_manifest(path, root=cfg.get('root'), source_pattern=cfg.get('source_pattern')))
        if not self.rows:
            raise ValueError('no paired samples')
        if len({r['pair_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate pair_id across manifests')
        self.sampler = AVTimeSampler(cfg.get('global_seconds', 6), cfg.get('local_seconds', 2), cfg.get('local_views', 4), training,
            cfg.get('short_policy', 'zero_pad'), cfg.get('minimum_seconds', .1))
        fps = float(cfg.get('video_fps', 8))
        if any(abs(cfg.get(f'video_frames_{mode}', default) - cfg.get(f'{mode}_seconds', seconds) * fps) > 1e-6
               for mode, default, seconds in (('global', 48, 6), ('local', 16, 2))):
            raise ValueError('video frame counts must match paired interval duration times target FPS')
        self.transforms = AVTransforms(cfg, training)
        self.decoder_factory = decoder_factory

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        random_state = random.getstate()
        seed = int(self.cfg.get('seed', 0)) + self.epoch * len(self.rows) + index
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                return self._load_pair(index)
        finally:
            random.setstate(random_state)

    def _load_pair(self, index):
        row = self.rows[index]
        try:
            decoder = self.decoder_factory(row, int(self.cfg.get('sample_rate', 16000)), self.cfg.get('sync', {}))
            global_interval, local_intervals = self.sampler(decoder.video_duration, decoder.audio_duration)
            global_view = self._decode_view(decoder, global_interval, local=False)
            locals_ = [self._decode_view(decoder, interval, local=True) for interval in local_intervals]
            all_views = [global_view, *locals_]
            diagnostics = dict(decoder.diagnostics,
                padded=any(bool(v['video_padding_mask'].any() or v['audio_padding_mask'].any()) for v in all_views),
                valid_global_seconds=global_interval[1] - global_interval[0],
                max_frame_sampling_drift_seconds=max(float(v.get('video_sampling_drift_seconds', 0)) for v in all_views),
                max_audio_boundary_rounding_seconds=max(float(v.get('audio_rounding_delta_seconds', torch.zeros(2)).abs().max()) for v in all_views))
            return {'global': global_view, 'local': locals_, **{k: row[k] for k in ('pair_id', 'source_id', 'video_path', 'audio_path')},
                    'sync_diagnostics': diagnostics}
        except Exception as error:
            if self.on_decode_error == 'skip':
                return {'decode_error': {**{key: row[key] for key in
                    ('pair_id', 'source_id', 'video_path', 'audio_path')},
                    'reason': f'{type(error).__name__}: {error}'}}
            raise RuntimeError(f'paired decode failed pair_id={row["pair_id"]} source_id={row["source_id"]}: {error}') from error

    def _decode_view(self, decoder, interval, local=False):
        import math
        import torch.nn.functional as F
        mode = 'local' if local else 'global'
        seconds = float(self.cfg.get(f'{mode}_seconds', 2 if local else 6))
        target_frames = int(self.cfg.get(f'video_frames_{mode}', 16 if local else 48))
        duration = interval[1] - interval[0]
        # Short clips retain their real time axis and are never stretched/repeated.
        frames = min(target_frames, max(1, math.floor(duration * float(self.cfg.get('video_fps', 8)) + 1e-8)))
        view = self.transforms(decoder.decode(interval, frames), local=local)
        length = int(view['audio_lengths'])
        target_samples = round(seconds * int(self.cfg.get('sample_rate', 16000)))
        if length > target_samples:
            raise ValueError('decoded audio exceeds configured view length')
        # Pad AFTER augmentation/normalization, so padding is exactly zero.
        view['video'] = F.pad(view['video'], (0, 0, 0, 0, 0, target_frames - frames))
        view['audio'] = F.pad(view['audio'], (0, target_samples - length))
        view['video_lengths'] = torch.tensor(frames)
        view['video_padding_mask'] = torch.arange(target_frames) >= frames
        view['audio_padding_mask'] = torch.arange(target_samples) >= length
        for key in ('video_frame_times', 'video_frame_indices'):
            view[key] = F.pad(view[key], (0, target_frames - frames))
        return view


def make_va_lejepa_loader(cfg, rank=0, world_size=1):
    dataset = VALeJEPADataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg.get('seed', 0)))
    loader = DataLoader(dataset, batch_size=int(cfg.get('batch_size', 1)), sampler=sampler,
        num_workers=int(cfg.get('num_workers', 4)), pin_memory=True, drop_last=True, collate_fn=collate_va_lejepa)
    return dataset, loader, sampler
