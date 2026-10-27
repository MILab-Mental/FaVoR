"""Frame encoders and native video encoders keep distinct readouts and inputs."""
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode

from BASELINES.common.official import official
from BASELINES.common.runtime import load_file, path, strict_load
from .audio import hf_load

IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)


def frames_tensor(video, size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, training=False, square_resize=False):
    # One spatial crop/flip shared by every frame preserves temporal consistency.
    frames = video.permute(0, 3, 1, 2).float() / 255
    frames = TF.resize(frames, [size, size] if square_resize else size,
                       interpolation=InterpolationMode.BICUBIC, antialias=True)
    if training:
        from torchvision.transforms import RandomCrop
        top, left, height, width = RandomCrop.get_params(frames, (size, size))
        frames = TF.crop(frames, top, left, height, width)
        if torch.rand(()) < .5:
            frames = TF.hflip(frames)
    else:
        frames = TF.center_crop(frames, [size, size])
    return TF.normalize(frames, mean, std)


class FrameEncoder(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        self.kind = cfg['kind']
        if self.kind == 'dinov2':
            with official('dinov2') as source:
                self.encoder = source('dinov2.hub.backbones').dinov2_vitl14(pretrained=False)
            if initialize:
                strict_load(self.encoder, load_file(cfg['weights']))
            self.output_dim = 1024
            self.processor = None
        else:
            from transformers import AutoImageProcessor, AutoConfig, AutoModel, CLIPVisionModelWithProjection, CLIPVisionConfig
            source_path = str(path(cfg['weights'])) if cfg.get('weights') else cfg['hub_id']
            self.processor = AutoImageProcessor.from_pretrained(source_path, revision=cfg['revision'])
            cls = CLIPVisionModelWithProjection if self.kind == 'clip' else AutoModel
            if initialize:
                self.encoder = hf_load(cls, cfg)
            else:
                config = (CLIPVisionConfig if self.kind == 'clip' else AutoConfig).from_pretrained(source_path, revision=cfg['revision'])
                self.encoder = cls(config) if self.kind == 'clip' else AutoModel.from_config(config)
            self.output_dim = self.encoder.config.projection_dim if self.kind == 'clip' else self.encoder.config.hidden_size
        self.input_spec = dict(modality='video', seconds=float(cfg.get('seconds', 4)), frames=int(cfg.get('frames', 8)), size=224)
        self.frame_batch = int(cfg.get('frame_batch', 4))

    def prepare(self, clip):
        if self.processor is None:
            frames = frames_tensor(clip['video'], training=self.training)
        else:
            frames = self.processor(images=list(clip['video'].numpy()), return_tensors='pt')['pixel_values']
        return dict(pixel_values=frames)

    def forward(self, batch):
        features = []
        for frames in batch['pixel_values'].split(self.frame_batch):
            if self.kind == 'dinov2':
                values = self.encoder.forward_features(frames)['x_norm_clstoken']
            else:
                result = self.encoder(pixel_values=frames)
                values = result.image_embeds if self.kind == 'clip' else result.pooler_output
            features.append(values)
        return torch.cat(features).mean(0, keepdim=True)


class VideoMAE(nn.Module):
    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        self.pretraining = pretrain
        self.output_dim = 1024
        self.mask_ratio = float(cfg.get('mask_ratio', .9))
        with official('VideoMAE') as source:
            if pretrain:
                self.encoder = source('modeling_pretrain').pretrain_videomae_large_patch16_224(decoder_depth=12, use_checkpoint=True)
            else:
                self.encoder = source('modeling_finetune').vit_large_patch16_224(num_classes=1, all_frames=16, use_checkpoint=True)
                self.encoder.head = nn.Identity()
            self.mask_generator = source('masking_generator').TubeMaskingGenerator((8, 14, 14), self.mask_ratio)
        if initialize:
            checkpoint = load_file(cfg['weights'])
            state = checkpoint.get('encoder', checkpoint.get('model', checkpoint))
            if pretrain:
                strict_load(self.encoder, state)
            else:
                if any(k.startswith('encoder.') for k in state):
                    state = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
                strict_load(self.encoder, state, discard=('head.', 'norm.'), allow_missing=('fc_norm.',))
        self.input_spec = dict(modality='video', seconds=16 * 4 / 30, frames=16, size=224, frame_stride=4)

    def prepare(self, clip):
        video = frames_tensor(clip['video'], training=self.training).permute(1, 0, 2, 3)[None]
        return dict(video=video)

    def forward(self, batch):
        return self.encoder.forward_features(batch['video'])

    def pretrain_loss(self, batches, step=0):
        from einops import rearrange
        videos = torch.cat([b['video'] for b in batches])
        mask = torch.as_tensor(np.stack([self.mask_generator() for _ in batches]), device=videos.device, dtype=torch.bool)
        mean = videos.new_tensor(IMAGENET_MEAN)[None, :, None, None, None]
        std = videos.new_tensor(IMAGENET_STD)[None, :, None, None, None]
        pixels = rearrange(videos * std + mean, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=16, p2=16)
        pixels = (pixels-pixels.mean(-2, keepdim=True)) / (pixels.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        target = rearrange(pixels, 'b n p c -> b n (p c)')[mask].reshape(len(batches), -1, 1536)
        return F.mse_loss(self.encoder(videos, mask), target)

    def export_encoder(self):
        # Same mapping as official run_class_finetuning: replace final norm by
        # a freshly initialized post-mean fc_norm and discard decoder entirely.
        return {k: v for k, v in self.encoder.encoder.state_dict().items() if not k.startswith('norm.')}


class MAEDFER(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        with official('MAE-DFER') as source:
            self.encoder = source('modeling_finetune').vit_base_dim512_no_depth_patch16_160(
                depth=16, num_classes=1, all_frames=16, attn_type='local_global',
                lg_region_size=(2, 5, 10), lg_classify_token_type='region')
        self.encoder.head = nn.Identity()
        if initialize:
            checkpoint = load_file(cfg['weights'])
            state = checkpoint.get('model', checkpoint)
            state = {k.removeprefix('encoder.'): v for k, v in state.items()
                     if k.startswith('encoder.')}
            strict_load(self.encoder, state, discard=('head.', 'norm.'), allow_missing=('fc_norm.',))
        self.output_dim = 512
        self.input_spec = dict(modality='video', seconds=16 * 4 / 25, frames=16, size=160, frame_stride=4)

    def prepare(self, clip):
        return dict(video=frames_tensor(clip['video'], 160, training=self.training).permute(1, 0, 2, 3)[None])

    def forward(self, batch):
        return self.encoder.forward_features(batch['video'])


class VJEPA21(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        with official('vjepa2') as source:
            self.encoder = source('app.vjepa_2_1.models.vision_transformer').vit_large(
                img_size=(384, 384), num_frames=64, tubelet_size=2,
                use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
                use_activation_checkpointing=True)
        # Match the official final-layer feature extraction API during train
        # and eval; do not change readout when calling module.train().
        self.encoder.return_hierarchical = False
        if initialize:
            checkpoint = load_file(cfg['weights'])
            state = {k.removeprefix('module.').removeprefix('backbone.'): v for k, v in checkpoint['ema_encoder'].items()}
            strict_load(self.encoder, state)
        self.output_dim = 1024
        self.input_spec = dict(modality='video', seconds=64/24, frames=64, size=384)

    def prepare(self, clip):
        return dict(video=frames_tensor(clip['video'], 384, training=self.training).permute(1, 0, 2, 3)[None])

    def forward(self, batch):
        return self.encoder(batch['video'], training=False).mean(1)


class HieraVideo(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        with official('hiera') as source:
            self.encoder = source('hiera').hiera_large_16x224(pretrained=False, num_classes=0)
        self.encoder.head = nn.Identity()
        # The official video inference notebook uses .255 for its third
        # channel, unlike SlowFast's all-.225 default. Make this choice explicit.
        self.mean = tuple(cfg.get('mean', (.45, .45, .45)))
        self.std = tuple(cfg.get('std', (.225, .225, .255)))
        self.output_dim = self.encoder.norm.normalized_shape[0]
        if initialize:
            checkpoint = load_file(cfg['weights'])
            strict_load(self.encoder, checkpoint['model_state'],
                discard=('head.', 'decoder_', 'mask_token', 'multi_scale_fusion_heads.', 'encoder_norm.'),
                allow_missing=('norm.',))  # same fresh post-pooling norm as official encoder-only loading
        self.input_spec = dict(modality='video', seconds=16*4/30, frames=16, size=224, frame_stride=4,
                               mean=list(self.mean), std=list(self.std))

    def prepare(self, clip):
        return dict(video=frames_tensor(clip['video'], mean=self.mean, std=self.std,
                    training=self.training).permute(1, 0, 2, 3)[None])

    def forward(self, batch):
        return self.encoder(batch['video'])


class MaskFeat(nn.Module):
    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        from BASELINES.common.official import checkout
        self.pretraining = pretrain
        with official('SlowFast') as source, official('pytorchvideo'):
            config = source('slowfast.config.defaults').get_cfg()
            suffix = 'MaskFeat_PT' if pretrain else 'FT'
            config.merge_from_file(str(checkout('SlowFast') / f'configs/masked_ssl/k400_MVITv2_L_16x4_{suffix}.yaml'))
            config.NUM_GPUS = 0
            builder = source('slowfast.models.masked') if pretrain else source('slowfast.models.video_model_builder')
            self.encoder = builder.MaskMViT(config) if pretrain else builder.MViT(config)
            self.loss_fn = source('slowfast.models.losses').MultipleMSELoss() if pretrain else None
            self.mask_generator = source('slowfast.datasets.transform').MaskingGenerator3D(
                mask_window_size=config.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=round(np.prod(config.AUG.MASK_WINDOW_SIZE) * config.AUG.MASK_RATIO),
                max_num_patches=np.prod(config.AUG.MASK_WINDOW_SIZE[1:]),
                min_num_patches=np.prod(config.AUG.MASK_WINDOW_SIZE[1:])//5) if pretrain else None
        self.output_dim = 1152
        if not pretrain:
            self.encoder.head = nn.Identity()
        if initialize:
            checkpoint = load_file(cfg['weights'])
            self.load_encoder(checkpoint.get('encoder', checkpoint.get('model_state', checkpoint)))
        self.input_spec = dict(modality='video', seconds=16*4/30, frames=16, size=224)

    def prepare(self, clip):
        return dict(video=frames_tensor(clip['video'], mean=(.45,)*3, std=(.225,)*3,
                    training=self.training).permute(1, 0, 2, 3)[None])

    def forward(self, batch):
        return self.encoder([batch['video']])

    def pretrain_loss(self, batches, step=0):
        x = torch.cat([b['video'] for b in batches])
        masks = torch.as_tensor(np.stack([self.mask_generator() for _ in batches]), dtype=torch.float32, device=x.device)
        predictions, targets = self.encoder([x, None, masks])
        return self.loss_fn(predictions, targets)[0]

    def export_encoder(self):
        return {k: v for k, v in self.encoder.state_dict().items()
                if not k.startswith(('pred_head.', 'mask_token', 'hogs.'))}

    def load_encoder(self, state):
        import logging
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        target = self.encoder.state_dict()
        # The official FT config downsamples the last stage further than HOG
        # pretraining. Apply ONLY checkpoint.py's documented relative-position
        # interpolation; every other shape mismatch remains an error.
        for key in state.keys() & target.keys():
            if 'attn.rel_pos' in key and state[key].shape != target[key].shape:
                before = list(state[key].shape)
                state[key] = F.interpolate(state[key].T[None], size=target[key].shape[0], mode='linear')[0].T
                logging.getLogger('baselines').info('official relative-position interpolation %s %s -> %s', key, before, list(state[key].shape))
        return strict_load(self.encoder, state,
            discard=() if self.pretraining else ('pred_head.', 'mask_token', 'hogs.', 'head.'),
            allow_missing=() if self.pretraining else ('norm.',))
