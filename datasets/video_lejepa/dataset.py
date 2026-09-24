import logging
from pathlib import Path

import numpy as np
import torch

from datasets.common.weighted_sampler import DistributedWeightedSampler
from datasets.video_jepa.pretrain_dataset import VideoDataset

from .collate import collate_video_lejepa
from .views import VideoLeJEPAMultiCrop

logger = logging.getLogger(__name__)


class VideoLeJEPADataset(VideoDataset):
    """FAVOR manifest/decode dataset with a single shared temporal sample."""

    def __getitem__(self, index):
        original_index = int(index)
        for _ in range(10):
            path = self.samples[index]
            if isinstance(path, str) and path.lower().endswith((".jpg", ".jpeg", ".png")):
                loaded = self._load_image(index)
            else:
                loaded = self._load_video(index)
            if loaded is not None:
                loaded["path"] = str(path)
                return loaded
            index = int(np.random.randint(len(self)))
        raise RuntimeError(f"failed to decode a valid video after 10 attempts (started at {original_index})")

    def _load_video(self, index):
        dataset_index, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_index]
        frames, indices = self.loadvideo_decord(self.samples[index], frames_per_clip)
        if len(frames) == 0:
            return None
        views = self.transform(frames) if self.transform else {"frames": torch.as_tensor(frames)}
        views["frame_indices"] = torch.as_tensor(indices[0], dtype=torch.long)
        return views

    def _load_image(self, index):
        import torchvision

        dataset_index, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_index]
        try:
            image = torchvision.io.read_image(self.samples[index]).permute(1, 2, 0)
        except Exception:
            return None
        frames = image.unsqueeze(0).repeat(frames_per_clip, 1, 1, 1)
        views = self.transform(frames)
        views["frame_indices"] = torch.zeros(frames_per_clip, dtype=torch.long)
        return views


def make_video_lejepa_dataset(data_cfg, rank=0, world_size=1, training=True):
    paths = data_cfg.get("datasets", [])
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("data.datasets must contain at least one manifest")
    configured_frames = data_cfg.get("dataset_fpcs")
    num_frames = int(data_cfg.get("num_frames", configured_frames[0] if configured_frames else 16))
    if configured_frames is None:
        configured_frames = [num_frames] * len(paths)
    if len(configured_frames) != len(paths):
        raise ValueError("data.dataset_fpcs must have one value per data.datasets entry")
    if any(int(value) != num_frames for value in configured_frames):
        raise ValueError(
            "VIDEO-LeJEPA batches require a single temporal shape: all dataset_fpcs "
            f"must equal data.num_frames={num_frames}, got {configured_frames}"
        )

    fps = data_cfg.get("fps")
    frame_stride = data_cfg.get("frame_stride", data_cfg.get("frame_step"))
    if fps is not None and frame_stride is not None:
        raise ValueError("configure exactly one of data.fps or data.frame_stride/frame_step")
    if fps is None and frame_stride is None:
        frame_stride = 4

    transform = VideoLeJEPAMultiCrop(
        global_size=data_cfg.get("global_size", data_cfg.get("crop_size", 224)),
        local_size=data_cfg.get("local_size", 96),
        local_views=data_cfg.get("local_views", 10),
        global_scale=data_cfg.get("global_scale", (0.8, 1.0)),
        local_scale=data_cfg.get("local_scale", (0.02, 0.4)),
        local_aspect_ratio=data_cfg.get("local_aspect_ratio", (0.75, 1.3333333333333333)),
        horizontal_flip=data_cfg.get("horizontal_flip", True) if training else False,
        color_jitter_prob=data_cfg.get("color_jitter_prob", 0.8) if training else 0.0,
        grayscale_prob=data_cfg.get("grayscale_prob", 0.2) if training else 0.0,
        gaussian_blur_prob=data_cfg.get("gaussian_blur_prob", 0.0) if training else 0.0,
        color_jitter_hue=data_cfg.get("color_jitter_hue", 0.1),
    )
    dataset = VideoLeJEPADataset(
        data_paths=list(paths),
        datasets_weights=data_cfg.get("datasets_weights"),
        dataset_fpcs=[num_frames] * len(paths),
        fps=fps,
        frame_step=frame_stride,
        num_clips=1,
        transform=transform,
        random_clip_sampling=training,
        allow_clip_overlap=False,
        filter_short_videos=data_cfg.get("filter_short_videos", False),
    )
    if data_cfg.get("datasets_weights") is not None:
        sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=training
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        sampler=sampler,
        drop_last=bool(data_cfg.get("drop_last", training)),
        pin_memory=bool(data_cfg.get("pin_mem", True)),
        num_workers=int(data_cfg.get("num_workers", 1)),
        persistent_workers=int(data_cfg.get("num_workers", 1)) > 0 and bool(data_cfg.get("persistent_workers", True)),
        collate_fn=collate_video_lejepa,
    )
    logger.info("VIDEO-LeJEPA dataset: %d samples, %d frames, %d local views", len(dataset), num_frames, transform.local_views)
    return dataset, loader, sampler
