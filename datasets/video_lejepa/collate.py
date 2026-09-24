import torch


def collate_video_lejepa(samples):
    if not samples:
        raise ValueError("cannot collate an empty VIDEO-LeJEPA batch")
    return {
        "global_video": torch.stack([sample["global_video"] for sample in samples]),
        "local_video": torch.stack([sample["local_video"] for sample in samples]),
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
        "path": [sample["path"] for sample in samples],
    }
