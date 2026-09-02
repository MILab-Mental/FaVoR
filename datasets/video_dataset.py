"""Compatibility exports for the legacy combined video dataset module.

New code should import pre-training and fine-tuning datasets from
``datasets.video_pretrain_dataset`` and ``datasets.video_finetune_dataset``.
"""

from datasets.video_finetune_dataset import VideoCSVDataset, make_videodataset_finetune_v
from datasets.video_pretrain_dataset import VideoDataset, make_videodataset

__all__ = [
    "VideoDataset",
    "make_videodataset",
    "VideoCSVDataset",
    "make_videodataset_finetune_v",
]
