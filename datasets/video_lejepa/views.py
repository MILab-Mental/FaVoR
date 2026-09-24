import torch
from torchvision.transforms import v2


class VideoLeJEPAMultiCrop:
    """Create one clean global and K strong local views from the same frames."""

    def __init__(
        self,
        global_size=224,
        local_size=96,
        local_views=10,
        global_scale=(0.8, 1.0),
        local_scale=(0.02, 0.4),
        local_aspect_ratio=(0.75, 1.3333333333333333),
        horizontal_flip=True,
        color_jitter_prob=0.8,
        grayscale_prob=0.2,
        gaussian_blur_prob=0.0,
        color_jitter_hue=0.1,
        normalize=True,
    ):
        self.local_views = int(local_views)
        if self.local_views < 1:
            raise ValueError("local_views must be at least one")
        tail = []
        if normalize:
            tail = [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        self.global_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    (int(global_size), int(global_size)),
                    scale=tuple(global_scale),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                *tail,
            ]
        )
        strong = []
        if horizontal_flip:
            strong.append(v2.RandomHorizontalFlip(0.5))
        if color_jitter_prob > 0:
            strong.append(
                v2.RandomApply(
                    [v2.ColorJitter(0.4, 0.4, 0.2, color_jitter_hue)],
                    p=float(color_jitter_prob),
                )
            )
        if grayscale_prob > 0:
            strong.append(v2.RandomGrayscale(float(grayscale_prob)))
        if gaussian_blur_prob > 0:
            strong.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=float(gaussian_blur_prob),
                )
            )
        self.local_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    (int(local_size), int(local_size)),
                    scale=tuple(local_scale),
                    ratio=tuple(local_aspect_ratio),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                *strong,
                *tail,
            ]
        )

    def __call__(self, frames):
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        if frames.ndim != 4:
            raise ValueError(f"expected [T,H,W,C] or [T,C,H,W], got {tuple(frames.shape)}")
        if frames.shape[-1] in (1, 3, 4):
            frames = frames.permute(0, 3, 1, 2)
        if frames.shape[1] == 4:
            frames = frames[:, :3]
        global_video = self.global_transform(frames).permute(1, 0, 2, 3).contiguous()
        local_video = torch.stack(
            [self.local_transform(frames).permute(1, 0, 2, 3) for _ in range(self.local_views)]
        ).contiguous()
        return {"global_video": global_video, "local_video": local_video}
