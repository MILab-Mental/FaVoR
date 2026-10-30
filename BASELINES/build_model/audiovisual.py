"""External joint audio-video backbones."""
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchvision.transforms import v2

from BASELINES.common.official import official
from BASELINES.common.runtime import load_file, path, strict_load
from .audio import fbank
from .video import frames_tensor, IMAGENET_MEAN, IMAGENET_STD


def joint_only(model):
    """Prune norms used only by the official unimodal inference branches."""
    removed = []
    for name, module in model.named_modules():
        for key in ('norm_a', 'norm_v', 'norm1_a', 'norm1_v', 'norm2_a', 'norm2_v'):
            if hasattr(module, key):
                removed.append((name + '.' if name else '') + key + '.')
                setattr(module, key, nn.Identity())
    return tuple(removed)


def joint_tokens(model, batch):
    """The official multimodal fine-tuning path shared by CAV-MAE/Social-MAE."""
    audio = model.patch_embed_a(batch['audio'].unsqueeze(1).transpose(2, 3)) + model.pos_embed_a + model.modality_a
    video = model.patch_embed_v(batch['video']) + model.pos_embed_v + model.modality_v
    for block in model.blocks_a:
        audio = block(audio)
    for block in model.blocks_v:
        video = block(video)
    x = torch.cat([audio, video], 1)
    for block in model.blocks_u:
        x = block(x)
    return model.norm(x), video.shape[1]


def global_contrastive(encoder):
    """Keep the native loss, with both ranks contributing real negative pairs."""
    native=encoder.forward_contrastive
    def contrastive(audio,video,*args,**kwargs):
        from BASELINES.common import distributed
        if distributed.active():
            from torch.distributed.nn.functional import all_gather
            audio=torch.cat(all_gather(audio))
            video=torch.cat(all_gather(video))
        return native(audio,video,*args,**kwargs)
    encoder.forward_contrastive=contrastive


class CAVMAE(nn.Module):
    def gradient_components(self):
        return dict(audio_input=self.encoder.patch_embed_a, video_input=self.encoder.patch_embed_v)

    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        self.pretraining = pretrain
        with official('cav-mae', 'src') as source:
            module = source('models.cav_mae')
            self.encoder = module.CAVMAE(norm_pix_loss=True) if pretrain else module.CAVMAEFT(label_dim=1)
        removed = ()
        if not pretrain:
            self.encoder.mlp_head = nn.Identity()
            removed = joint_only(self.encoder)
        if initialize:
            checkpoint = load_file(cfg['weights'])
            state = checkpoint.get('encoder', checkpoint.get('model', checkpoint))
            strict_load(self.encoder, state, discard=() if pretrain else
                        ('decoder_', 'mask_token', 'mlp_head.') + removed)
        self.output_dim = 768
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=10.24, frames=1, size=224)
        if pretrain:
            self.input_spec['minimum_batch_size'] = 2
            global_contrastive(self.encoder)

    def prepare(self, clip):
        audio, valid = fbank(clip['audio'], mean=-5.081, std=4.4849)
        video = frames_tensor(clip['video'], training=self.training)
        return dict(audio=audio[None], video=video, valid=torch.tensor([valid]))

    def forward(self, batch):
        # Official multimodal FT path, with length-aware patch pooling.
        x, video_tokens = joint_tokens(self.encoder, batch)
        # Audio patch layout is [frequency, time] in CAV-MAE.
        mask = torch.arange(64, device=x.device).repeat(8)[None] < (batch['valid'][:, None]+15)//16
        mask = torch.cat([mask, torch.ones(x.shape[0], video_tokens, device=x.device, dtype=torch.bool)], 1)
        return (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    def pretrain_loss(self, batches, step=0):
        audio = torch.cat([b['audio'] for b in batches])
        video = torch.cat([b['video'] for b in batches])
        if self.training and len(batches) < 2:
            raise ValueError('CAV-MAE contrastive training requires at least two paired samples')
        return self.encoder(audio, video, mask_ratio_a=.75, mask_ratio_v=.75,
                            mae_loss_weight=1., contrast_loss_weight=.01)[0]

    def export_encoder(self):
        removed = ('decoder_', 'mask_token', 'norm_a.', 'norm_v.')
        return {k: v for k, v in self.encoder.state_dict().items()
                if not k.startswith(removed) and not any('.'+part+'.' in k for part in ('norm1_a', 'norm1_v', 'norm2_a', 'norm2_v'))}


class SocialMAE(CAVMAE):
    """Random native Social-MAE pretraining, then encoder-only transfer."""
    def __init__(self,cfg,initialize=True,pretrain=False):
        nn.Module.__init__(self)
        self.pretraining=pretrain
        with official('SocialMAE','src') as source:
            module=source('models.social_mae')
            constructor=module.SocialMAE if pretrain else module.SocialMAEFT
            kwargs={} if pretrain else dict(label_dim=1)
            self.encoder=constructor(n_frame=8,stride=2,tr_pos=True,**kwargs)
        if pretrain:global_contrastive(self.encoder)
        else:
            self.encoder.mlp_head=nn.Identity()
            joint_only(self.encoder)
        self.output_dim=768
        self.input_spec=dict(modality='audio-video',sample_rate=16000,seconds=4,
                             frames=8,fps=2,size=224)
        if pretrain:self.input_spec['minimum_batch_size']=2

    def prepare(self,clip):
        audio,valid=fbank(clip['audio'],mean=-6.2943,std=12.26344)
        # The release passes OpenCV BGR frames directly to its transform;
        # FAVOR decodes RGB. Preserve that channel order and bicubic resize.
        video=frames_tensor(clip['video'].flip(-1),training=False,square_resize=True).permute(1,2,3,0)[None]
        real_frames=video.shape[-1]
        video=video.repeat(1,1,1,1,(8+real_frames-1)//real_frames)[...,:8]
        return dict(audio=audio[None],video=video,valid=torch.tensor([valid]),
                    video_valid=torch.tensor([(real_frames+1)//2]))

    def forward(self,batch):
        x,_=joint_tokens(self.encoder,batch)
        audio=torch.arange(64,device=x.device).repeat(8)[None]<(batch['valid'][:,None]+15)//16
        video=torch.arange(4,device=x.device).repeat(196)[None]<batch['video_valid'][:,None]
        valid=torch.cat([audio,video],1)
        return (x*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)

    def gradient_components(self):
        modules=super().gradient_components()
        if self.pretraining:
            modules.update(audio_decoder=self.encoder.decoder_pred_a,video_decoder=self.encoder.decoder_pred_v,
                           audio_contrast=self.encoder.norm_a,video_contrast=self.encoder.norm_v)
        return modules


class LeAVJEPA(nn.Module):
    def gradient_components(self):
        return dict(audio_input=self.encoder.embedding.audio_patch_embed,
                    video_input=self.encoder.embedding.video_patch_embed)

    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        self.pretraining = pretrain
        with official('LeAVJEPA') as source:
            models = source('models')
            # Exact configs.py Large/8s configuration. Its import also imports
            # the unused WebDataset/TorchCodec loader; our raw-media decoder is
            # FAVOR's, so only the actual model configuration is needed here.
            audio_config = dict(spectrogram_size=(128, 801), patch_size=(16, 16), patch_stride=(16, 16), num_channels=1)
            video_config = dict(num_frames=16, tubelet_size=2, image_size=224, num_channels=3, patch_size=16)
            transformer_config = dict(hidden_size=1024, num_hidden_layers=24, intermediate_size=4096,
                num_attention_heads=16, attention_probs_dropout_prob=.1,
                hidden_dropout_prob=.1, qkv_bias=True, initializer_range=.02)
            self.encoder = models.Echo(audio_config, video_config, transformer_config, gradient_checkpointing=True)
            if pretrain:
                self.projector = models.Projector(in_dim=1024, hidden_dim=2048, out_dim=128)
                self.sigreg = models.SIGReg()
        if not pretrain:
            # EchoFineTuner normalizes its raw CLS token before supervision.
            # Keep this new downstream readout outside the exported SSL encoder.
            self.feature_norm = nn.LayerNorm(1024)
        if initialize:
            checkpoint = load_file(cfg['weights'])
            strict_load(self.encoder, checkpoint['encoder'])
        self.output_dim = 1024
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=8, frames=16, size=224)
        if pretrain:
            self.input_spec['pretrain_clips'] = 4
            self.input_spec['minimum_batch_size'] = 2
        self.spectrogram = nn.Sequential(torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128, n_fft=400, win_length=400,
            hop_length=160, window_fn=torch.hamming_window), torchaudio.transforms.AmplitudeToDB())
        self.global_transform = v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
            v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.local_transform = v2.Compose([v2.RandomResizedCrop(224, scale=(.4, 1.), antialias=True),
            v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def prepare(self, clip):
        # Match official 8-second Hamming log-mel padding and fixed normalization.
        if len(clip['video']) < 16:
            clip = dict(clip, video=torch.cat([clip['video'], clip['video'].new_zeros(16-len(clip['video']), *clip['video'].shape[1:])]))
        waveform = F.pad(clip['audio'], (0, max(0, 128000-len(clip['audio']))))[:128000]
        spec = (self.spectrogram.cpu()(waveform[None]) + 20.437003) / 24.496246
        frames = clip['video'].permute(0, 3, 1, 2)
        video = frames_tensor(clip['video'], training=False).permute(1, 0, 2, 3)[None]
        result = dict(video=video, audio=spec[None])
        if self.pretraining:
            # Official Large recipe: two joint global views and two unimodal
            # local views, dropping absent tokens rather than encoding zeros.
            result['global_video'] = torch.stack([self.global_transform(frames).permute(1, 0, 2, 3) for _ in range(2)])[None]
            result['local_video'] = self.local_transform(frames).permute(1, 0, 2, 3)[None]
        return result

    def forward(self, batch):
        features=self.encoder(batch['video'], batch['audio'])
        return features if self.pretraining else self.feature_norm(features)

    def prepare_views(self, clips):
        if len(clips) != 4:
            raise ValueError('LeAVJEPA requires G=2/K=2 independently sampled aligned AV crops')
        prepared = [self.prepare(clip) for clip in clips]
        return dict(global_video=torch.cat([p['global_video'][:, :1] for p in prepared[:2]], 1),
                    global_audio=torch.stack([p['audio'] for p in prepared[:2]], 1),
                    local_audio=prepared[2]['audio'], local_video=prepared[3]['local_video'])

    def pretrain_loss(self, batches, step=0):
        if self.training and len(batches) < 2:
            raise ValueError('LeAVJEPA projector BatchNorm/SIGReg requires at least two real samples')
        global_video = torch.cat([b['global_video'] for b in batches])
        global_audio = torch.cat([b['global_audio'] for b in batches])
        audio = torch.cat([b['local_audio'] for b in batches])
        local_video = torch.cat([b['local_video'] for b in batches])
        batch_size = len(batches)
        globals_cls = self.encoder(global_video.flatten(0, 1), global_audio.flatten(0, 1))
        z_global = self.projector(globals_cls).reshape(batch_size, 2, -1).transpose(0, 1)
        local_cls = torch.stack([self.encoder(None, audio), self.encoder(local_video, None)], 1)
        z_local = self.projector(local_cls.flatten(0, 1)).reshape(batch_size, 2, -1).transpose(0, 1)
        all_views = torch.cat([z_global, z_local])
        invariance = (z_global.mean(0) - all_views).square().mean()
        regularizer = torch.stack([self.sigreg(z, step) for z in all_views]).mean()
        return .95 * invariance + .05 * regularizer

    def export_encoder(self):
        return self.encoder.state_dict()


class LanguageBind(nn.Module):
    def gradient_components(self):
        return {mode+'_'+part:module for mode,branch in self.encoder.items() for part,module in branch.items()}

    def counted_components(self):
        return {mode+'_encoder_and_projection':branch for mode,branch in self.encoder.items()}

    def __init__(self, cfg, initialize=True):
        super().__init__()
        from .audio import hf_load
        self.encoder = nn.ModuleDict()
        self.transforms = {}
        with official('LanguageBind') as source:
            package = source('languagebind')
            for mode in ('audio', 'video'):
                cls = getattr(package, 'LanguageBind'+mode.title())
                model = hf_load(cls, cfg[mode])
                if getattr(model.config.vision_config, 'add_lora', False):
                    raise ValueError('LanguageBind requires the full FT release, not LoRA weights')
                self.encoder[mode] = nn.ModuleDict(dict(backbone=model.vision_model, projection=model.visual_projection))
                if mode == 'audio':
                    self.transforms[mode] = source('languagebind.audio.processing_audio').get_audio_transform(model.config)
                    self.sample_rate = model.config.vision_config.audio_sample_rate
                else:
                    frames = model.config.vision_config.num_frames
                dim = model.config.projection_dim
        self.output_dim = 2*dim
        # Its unused video-file helper installs a global Torch decord bridge;
        # FAVOR's actual paired decoder requires decord's native NDArray API.
        import decord
        decord.bridge.set_bridge('native')
        self.input_spec = dict(modality='audio-video', sample_rate=self.sample_rate, seconds=10, frames=frames, size=224)

    def prepare(self, clip):
        audio = self.transforms['audio']((clip['audio'][None].clone(), self.sample_rate))[None]
        frames=clip['video']
        if len(frames)!=self.input_spec['frames']:
            frames=frames[torch.linspace(0,len(frames)-1,self.input_spec['frames']).long()]
        video = frames_tensor(frames, mean=(.48145466, .4578275, .40821073),
                              std=(.26862954, .26130258, .27577711), training=self.training).permute(1, 0, 2, 3)[None]
        return dict(audio=audio, video=video)

    def forward(self, batch):
        features = []
        for mode in ('audio', 'video'):
            x = self.encoder[mode]['backbone'](pixel_values=batch[mode])[1]
            x = self.encoder[mode]['projection'](x)
            features.append(F.normalize(x, dim=-1))
        return torch.cat(features, -1)
