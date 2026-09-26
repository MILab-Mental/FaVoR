import torch
from torchvision.transforms import v2
from datasets.video_lejepa import VideoLeJEPAMultiCrop
from datasets.audio_lejepa.augment import augment_local_waveform
from datasets.audio_jepa.pretrain_dataset import normalize_waveform


class AVTransforms:
    def __init__(self, cfg, training=True):
        self.training = training
        self.normalize_audio = cfg.get('normalize_audio', True)
        self.audio_augmentation = cfg.get('audio_augmentation', {})
        if self.audio_augmentation.get('speed', {}).get('enabled', False):
            raise ValueError('speed augmentation changes physical token times and is disallowed for strict VA alignment')
        aug = dict(cfg.get('video_augmentation', {}))
        self.spatial = VideoLeJEPAMultiCrop(global_size=cfg.get('video_global_size', 112),
            local_size=cfg.get('video_local_size', 48), local_views=max(1, cfg.get('local_views', 4)), **aug)
        self.eval_transform = v2.Compose([v2.Resize(int(cfg.get('video_global_size', 112)), antialias=True),
            v2.CenterCrop(int(cfg.get('video_global_size', 112))), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __call__(self, view, local=False):
        frames = view['video'].permute(0, 3, 1, 2)
        transform = self.spatial.local_transform if local else self.spatial.global_transform
        if not self.training:
            transform = self.eval_transform
        view['video'] = transform(frames).permute(1, 0, 2, 3).contiguous()
        if local and self.training:
            view['audio'], _ = augment_local_waveform(view['audio'], self.audio_augmentation)
        if self.normalize_audio:
            view['audio'] = normalize_waveform(view['audio'])
        return view
