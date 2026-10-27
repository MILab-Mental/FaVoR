"""Joint AV backbones. CAV-MAE uses image frames; SocialMAE uses video tubelets."""
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
    """The shared, unchanged multimodal FT path of CAV-MAE and SocialMAE."""
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


class SocialMAE(nn.Module):
    def gradient_components(self):
        return dict(audio_input=self.encoder.patch_embed_a, video_input=self.encoder.patch_embed_v)

    def __init__(self, cfg, initialize=True):
        super().__init__()
        with official('SocialMAE', 'src') as source:
            self.encoder = source('models.social_mae').SocialMAEFT(label_dim=1, n_frame=8, stride=2)
        self.encoder.mlp_head = nn.Identity()
        removed = joint_only(self.encoder)
        if initialize:
            checkpoint = load_file(cfg['weights'])
            strict_load(self.encoder, checkpoint.get('model', checkpoint),
                        discard=('decoder_', 'mask_token', 'mlp_head.') + removed)
        self.output_dim = 768
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=4, frames=8, size=224)

    def prepare(self, clip):
        audio, valid = fbank(clip['audio'], mean=-6.2943, std=12.26344)
        # SocialMAE's Conv3d expects C,H,W,T, unlike other video backbones.
        video = frames_tensor(clip['video'], training=self.training, square_resize=True).permute(1, 2, 3, 0)[None]
        real_frames = video.shape[-1]
        video = F.pad(video, (0, 8-real_frames))
        return dict(audio=audio[None], video=video, valid=torch.tensor([valid]),
                    video_valid=torch.tensor([(real_frames+1)//2]))

    def forward(self, batch):
        x, _ = joint_tokens(self.encoder, batch)
        valid = torch.arange(64, device=x.device).repeat(8)[None] < (batch['valid'][:, None]+15)//16
        visual_valid = torch.arange(4, device=x.device).repeat(196)[None] < batch['video_valid'][:, None]
        valid = torch.cat([valid, visual_valid], 1)
        return (x*valid[..., None]).sum(1) / valid.sum(1, keepdim=True)


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
        return self.encoder(batch['video'], batch['audio'])

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


class AVHubert(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        import fairseq
        with official('av_hubert') as source:
            source('avhubert')
            models, checkpoint_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [str(path(cfg['weights']))], strict=True)
            self.encoder = models[0]
        self.encoder.remove_pretraining_modules()
        # The label embedding table is used only for the upstream cluster loss.
        self.encoder.label_embs_concat = None
        self.encoder.mask_emb = None
        self.encoder.feature_grad_mult = 1.0
        self.output_dim = self.encoder.encoder_embed_dim
        if self.output_dim != 1024:
            raise ValueError('AV-HuBERT requires Large clean PT')
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=4,
                              frames=100, fps=25, size=88, requires_lips=True)

    def prepare(self, clip):
        from python_speech_features import logfbank
        from torchvision.transforms import functional as TF
        import numpy as np
        # Official logfbank uses PCM16 amplitude, then stacks four 100 Hz frames.
        waveform = (clip['audio'].numpy() * 32768).clip(-32768, 32767).astype(np.int16)
        audio = torch.from_numpy(logfbank(waveform, samplerate=16000).astype('float32'))
        audio = F.pad(audio, (0, 0, 0, (-len(audio)) % 4)).reshape(-1, 104)
        audio = F.layer_norm(audio, audio.shape[1:])
        frames = clip['video'].permute(0, 3, 1, 2).float() / 255
        if min(frames.shape[-2:]) < 88:
            raise ValueError('Lip ROI must be at least 88x88; do not pad a missing crop')
        video = (TF.center_crop(TF.rgb_to_grayscale(frames), [88, 88])-.421)/.165
        length = min(len(audio), len(video))
        return dict(audio=audio[:length].T[None], video=video[:length].permute(1, 0, 2, 3)[None],
                    padding_mask=torch.zeros(1, length, dtype=torch.bool))

    def forward(self, batch):
        x, mask = self.encoder.extract_features(source={k: batch[k] for k in ('audio', 'video')},
                                                padding_mask=batch['padding_mask'], mask=False)
        valid = (~mask)[..., None] if mask is not None else torch.ones_like(x[..., :1])
        return (x*valid).sum(1) / valid.sum(1).clamp_min(1)


class LanguageBind(nn.Module):
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
        self.input_spec = dict(modality='audio-video', sample_rate=self.sample_rate, seconds=10, frames=frames, size=224)

    def prepare(self, clip):
        audio = self.transforms['audio']((clip['audio'][None].clone(), self.sample_rate))[None]
        video = frames_tensor(clip['video'], mean=(.48145466, .4578275, .40821073),
                              std=(.26862954, .26130258, .27577711), training=self.training).permute(1, 0, 2, 3)[None]
        return dict(audio=audio, video=video)

    def forward(self, batch):
        features = []
        for mode in ('audio', 'video'):
            x = self.encoder[mode]['backbone'](pixel_values=batch[mode])[1]
            x = self.encoder[mode]['projection'](x)
            features.append(F.normalize(x, dim=-1))
        return torch.cat(features, -1)


class FavorNoJoint(nn.Module):
    """Reuse the exact FAVOR downstream structure and separate A/V initialization."""
    def __init__(self, cfg, initialize=True):
        super().__init__()
        from BASELINES.common.data import configure_instances
        configure_instances(cfg, 'audio-video')
        from app.main import load_config
        from models.va_lejepa import build_va_lejepa
        from models.va_lejepa.finetune import VAFinetuner
        from BASELINES.common.runtime import path
        native = load_config(path(cfg['model']['favor_config']))
        joint = load_config(path(cfg['model']['favor_pretrain_config']))
        initialization_keys = {'init_source', 'weight_source', 'vjepa_checkpoint', 'video_lejepa_checkpoint',
                               'emotion2vec_checkpoint', 'audio_lejepa_checkpoint'}
        for mode in ('audio', 'video'):
            left, right = native['model'][mode], joint['model'][mode]
            structural = (left.keys() | right.keys()) - initialization_keys - {'freeze'}
            if any(left.get(k) != right.get(k) for k in structural):
                raise ValueError(f'P0 {mode} structure differs from the joint SSL counterpart')
            for key in initialization_keys:
                left.pop(key, None)
                if key in right:
                    left[key] = right[key]
        if native['model']['fusion'] != joint['model']['fusion']:
            raise ValueError('P0 fusion structure differs from the joint SSL counterpart')
        cfg['p0_joint_counterpart'] = cfg['model']['favor_pretrain_config']
        cfg['p0_initialization'] = {mode:{k:v for k,v in native['model'][mode].items() if k in initialization_keys}
                                   for mode in ('audio', 'video')}
        cfg['p0_optimizer_scales'] = {k:v for k,v in native['finetune'].items() if k.endswith('_lr_scale')}
        data = native['data']
        if native.get('finetune', {}).get('freeze_video_encoder') or native.get('finetune', {}).get('freeze_audio_encoder'):
            raise ValueError('The P0 no-joint control must train both backbones')
        va = build_va_lejepa(native['model'], data, initialize=initialize)
        outputs = 1 if cfg['data']['task'] == 'regression' else int(cfg['data']['num_class'])
        finetune = dict(native['finetune'])
        self.native_finetune = finetune
        finetune['pretrained_checkpoint'] = None
        if 'instance_aggregator' in native['model']:
            finetune['instance_aggregator'] = native['model']['instance_aggregator']
        if cfg['model'].get('instance_aggregator'):
            finetune['instance_aggregator'] = cfg['model']['instance_aggregator']
        self.encoder = VAFinetuner(va, outputs, finetune)
        from datasets.va_lejepa.transforms import AVTransforms
        self.transforms = AVTransforms(data, training=True)
        self.input_spec = dict(modality='audio-video', sample_rate=data['sample_rate'],
            seconds=data['global_seconds'], frames=data['video_frames_global'], size=data['video_global_size'],
            favor_data_cfg=data)

    def prepare(self, clip):
        if clip.get('_favor_prepared') is not True:
            raise ValueError('P0 must use the original FAVOR view decoder/transforms')
        view = clip
        return dict(global_view={k: v[None] for k, v in view.items() if torch.is_tensor(v)})

    def forward(self, batch):
        return self.encoder({'global': batch['global_view']})

    def predict_sample(self, sample, task, device):
        from datasets.va_lejepa.collate import collate_views, to_device
        from utils.instance_evaluation import average_window_outputs
        prepared = [clips[0] for clips in sample['sentences']]
        outputs = []
        for indices in sample['windows']:
            # Reuse native VAFinetuner's sentence aggregator and branch heads.
            views = [{k: v for k, v in prepared[i].items() if torch.is_tensor(v)} for i in indices]
            view = collate_views(views)
            if self.encoder.instance_enabled:
                view = {k: v[None] for k, v in view.items()}
                outputs.append(self.encoder({'global': to_device(view, device)}))
            else:
                result = self.encoder({'global': to_device(view, device)})
                sentences = [dict(logits=result['logits'][i:i+1],
                    branches={k:v[i:i+1] for k,v in result['branches'].items()}) for i in range(len(indices))]
                outputs.append(average_window_outputs(sentences, task))
        return average_window_outputs(outputs, task)

    def make_optimizer(self, cfg):
        from app.pretrain_va_lejepa.train import _optimizer
        opt = dict(cfg['optimization'])
        for family, scale in (('video_pretrained', 'video'), ('audio_pretrained', 'audio'), ('new', 'fusion'), ('head', 'head')):
            opt['lr_'+family] = float(opt['lr']) * float(self.native_finetune.get(scale+'_lr_scale', 1))
        return _optimizer(self.encoder, opt)

    def supervised_loss(self, output, labels, criterion, task):
        from app.finetune_va.train import _loss
        weight = float(self.native_finetune.get('branch_loss_weight', .1 if self.encoder.strategy=='transformer' else 0))
        return _loss(output, labels, criterion, task, weight)

    def gradient_components(self):
        return dict(audio_backbone=self.encoder.va.audio_encoder, video_backbone=self.encoder.va.video_encoder,
                    fusion=self.encoder.va.fusion_adapters, task_heads=self.encoder.heads)

    def counted_components(self):
        return dict(favor_audio_backbone=self.encoder.va.audio_encoder,
                    favor_video_backbone=self.encoder.va.video_encoder,
                    favor_full_downstream_including_heads=self.encoder)


class AVSiam(nn.Module):
    """Published Base+ only; two shared streams followed by native fusion blocks."""
    def __init__(self, cfg, initialize=True):
        super().__init__()
        import ast
        import sys
        import types
        import timm
        from BASELINES.common.official import checkout
        directory = checkout('AVSiam') / 'src/models'
        filename = directory / 'cav_mae_base.py'
        tree = ast.parse(filename.read_text())
        # The release imports debugging/ToMe files that are absent, but their
        # only call sites are commented out. Verify that before removing ONLY
        # these unused imports; execute all backbone/fusion code unchanged.
        unused = {'set_trace', 'bipartite_soft_matching', 'merge_source', 'merge_wavg', 'yb_bipartite_soft_matching'}
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in unused for n in ast.walk(tree)):
            raise ValueError('AVSiam now uses formerly dead imports; review the source adaptation')
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
                     node.module in ('ipdb', 'tome.merge', 'yb_tome'))]
        package_name = '_favor_baseline_avsiam'
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
        module = types.ModuleType(package_name + '.cav_mae_base')
        module.__file__, module.__package__ = str(filename), package_name
        sys.modules[module.__name__] = module
        with official('AVSiam', 'src'):
            exec(compile(tree, str(filename), 'exec'), module.__dict__)
            create_model = timm.create_model
            def no_redundant_download(name, **kwargs):
                if name != 'vit_base_patch16_224.augreg_in21k':
                    raise ValueError('Unexpected AVSiam backbone')
                kwargs['pretrained'] = False  # full Base+ checkpoint replaces these tensors
                return create_model(name, **kwargs)
            timm.create_model = no_redundant_download
            try:
                self.encoder = module.CAVMAEFT_BASE(label_dim=1)
            finally:
                timm.create_model = create_model
        for name in ('my_blocks', 'my_patch_embed', 'my_patch_embed_a'):
            delattr(self.encoder, name)
        for name in ('mlp_head', 'mlp_head_a', 'mlp_head_mm', 'mlp_head_mm_v2'):
            setattr(self.encoder, name, nn.Identity())
        self.encoder.vit_base.head = nn.Identity()
        if initialize:
            from huggingface_hub import hf_hub_download
            weights = cfg.get('weights') or hf_hub_download(cfg['hub_id'], cfg['filename'], revision=cfg['revision'])
            checkpoint = load_file(weights)
            strict_load(self.encoder, checkpoint.get('model', checkpoint), discard=(
                'my_blocks.', 'my_patch_embed.', 'my_patch_embed_a.', 'ast_base.',
                'decoder_', 'mask_token', 'vit_base.head.', 'mlp_head', 'predictor.'))
        self.output_dim = 1536
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=10.24, frames=1, size=224)

    def prepare(self, clip):
        audio, _ = fbank(clip['audio'], mean=-5.081, std=4.4849)
        return dict(audio=audio[None], video=frames_tensor(clip['video'], training=self.training, square_resize=True)[None])

    def forward(self, batch):
        # Native training path returns fused, audio, video readouts. Only the
        # fused representation feeds our new task head in both train and eval.
        fused, _, _ = self.encoder(batch['audio'], batch['video'], mode='mm_grad', is_eval=False)
        return fused

    def gradient_components(self):
        return dict(audio_input=self.encoder.vit_base.patch_embed_a,
                    video_input=self.encoder.vit_base.patch_embed)
