"""Trainable audio encoders; upstream ASR heads/teachers never enter downstream counts."""
import logging

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from BASELINES.common.official import official
from BASELINES.common.runtime import load_file, path, strict_load


def fbank(waveform, frames=1024, mean=-4.2677393, std=9.1379948):
    waveform = waveform - waveform.mean()
    if waveform.numel() < 400:
        raise ValueError('Kaldi fbank requires at least 25 ms of real audio')
    features = torchaudio.compliance.kaldi.fbank(waveform[None], htk_compat=True,
        sample_frequency=16000, use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0., frame_shift=10)
    valid = min(len(features), frames)
    features = F.pad(features[:frames], (0, 0, 0, max(0, frames-len(features))))
    return (features - mean) / std, valid


def hf_load(cls, cfg, **kwargs):
    source = str(path(cfg['weights'])) if cfg.get('weights') else cfg['hub_id']
    if cfg.get('weights') and not path(cfg['weights']).is_dir():
        raise FileNotFoundError(f'{source}: download the fixed {cfg["hub_id"]}@{cfg["revision"]} snapshot first')
    model, report = cls.from_pretrained(source, revision=cfg['revision'],
        output_loading_info=True, **kwargs)
    allowed = tuple(cfg.get('discard_prefixes', []))
    unexpected = report.get('unexpected_keys', [])
    summary = dict(report, unexpected_keys=unexpected[:8], unexpected_count=len(unexpected))
    logging.getLogger('baselines').info('HF loading %s@%s: %s', cfg['hub_id'], cfg['revision'], summary)
    if report.get('missing_keys') or report.get('mismatched_keys') or report.get('error_msgs') or any(not k.startswith(allowed) for k in unexpected):
        raise ValueError(f'Unaccounted-for Hugging Face checkpoint keys: {report}')
    return model


class WaveformEncoder(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig, AutoFeatureExtractor
        source = str(path(cfg['weights'])) if cfg.get('weights') else cfg['hub_id']
        self.processor = AutoFeatureExtractor.from_pretrained(source, revision=cfg['revision'])
        self.encoder = hf_load(AutoModel, cfg) if initialize else AutoModel.from_config(AutoConfig.from_pretrained(source, revision=cfg['revision']))
        self.output_dim = self.encoder.config.hidden_size
        if self.output_dim != 1024 or self.encoder.config.num_hidden_layers != 24:
            raise ValueError('These waveform table entries require the specified 24-layer Large encoder')
        if cfg.get('gradient_checkpointing', True):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=float(cfg.get('seconds', 4)))

    def prepare(self, clip):
        return dict(self.processor(clip['audio'].numpy(), sampling_rate=16000, return_tensors='pt', return_attention_mask=True))

    def forward(self, batch):
        states = self.encoder(**batch).last_hidden_state
        lengths = self.encoder._get_feat_extract_output_lengths(batch['attention_mask'].sum(-1))
        mask = torch.arange(states.shape[1], device=states.device)[None] < lengths[:, None]
        return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class WhisperEncoder(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        from transformers import WhisperModel, WhisperConfig, WhisperFeatureExtractor
        source = str(path(cfg['weights'])) if cfg.get('weights') else cfg['hub_id']
        self.processor = WhisperFeatureExtractor.from_pretrained(source, revision=cfg['revision'])
        model = hf_load(WhisperModel, cfg) if initialize else WhisperModel(WhisperConfig.from_pretrained(source, revision=cfg['revision']))
        self.encoder = model.encoder
        self.output_dim = self.encoder.config.d_model
        if self.output_dim != 1280 or self.encoder.config.num_mel_bins != 128:
            raise ValueError('Whisper large-v3 requires original multilingual Whisper large-v3')
        if cfg.get('gradient_checkpointing', True):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=30)

    def prepare(self, clip):
        inputs = self.processor(clip['audio'].numpy(), sampling_rate=16000,
                                return_tensors='pt', return_attention_mask=True)
        return dict(input_features=inputs['input_features'], valid=(inputs['attention_mask'].sum(-1)+1)//2)

    def forward(self, batch):
        # Native Whisper attends over its fixed 30 s log-mel input. Readout
        # excludes padded time positions; its native attention protocol is retained.
        states = self.encoder(batch['input_features']).last_hidden_state
        mask = torch.arange(states.shape[1], device=states.device)[None] < batch['valid'][:, None]
        return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class AudioMAE(nn.Module):
    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        self.pretraining = pretrain
        self.output_dim = 1024 if cfg['size'] == 'large' else 768
        with official('AudioMAE') as source:
            if pretrain:
                # The public AudioMAE training recipe uses its patched local
                # Swin decoder. Import that file without altering installed timm.
                import importlib.util
                from BASELINES.common.official import checkout
                location = checkout('AudioMAE') / 'timm_patch/swin_transformer.py'
                module_spec = importlib.util.spec_from_file_location('timm.models._audiomae_swin', location)
                swin = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(swin)
                module = source('models_mae')
                module.SwinTransformerBlock = swin.SwinTransformerBlock
                self.encoder = getattr(module, f'mae_vit_{cfg["size"]}_patch16')(
                    img_size=(1024, 128), in_chans=1, audio_exp=True,
                    decoder_mode=1, norm_pix_loss=True, use_custom_patch=False)
            else:
                self.encoder = getattr(source('models_vit'), f'vit_{cfg["size"]}_patch16')(
                    img_size=(1024, 128), in_chans=1, num_classes=0, global_pool=True)
                self.encoder.pos_embed.requires_grad_(False)
        self.mask_ratio = float(cfg.get('mask_ratio', .8))
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=10.24)
        if initialize:
            checkpoint = load_file(cfg['weights'])
            state = checkpoint.get('encoder', checkpoint.get('model', checkpoint))
            strict_load(self.encoder, state, discard=() if pretrain else ('decoder_', 'mask_token', 'norm.'),
                        allow_missing=() if pretrain else ('fc_norm.',))

    def prepare(self, clip):
        features, valid = fbank(clip['audio'])
        return dict(features=features[None, None], valid=torch.tensor([valid]))

    def forward(self, batch):
        # Mean only genuine time patches, including the final partial patch.
        model = self.encoder
        x = model.patch_embed(batch['features']) + model.pos_embed[:, 1:]
        cls = (model.cls_token + model.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = model.pos_drop(torch.cat([cls, x], 1))
        for block in model.blocks:
            x = block(x)
        time = torch.arange(64, device=x.device).repeat_interleave(8)
        valid = time[None] < (batch['valid'][:, None] + 15) // 16
        return model.fc_norm((x[:, 1:] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1))

    def pretrain_loss(self, batches, step=0):
        model = self.encoder
        x = torch.cat([b['features'] for b in batches])
        latent, mask, restore, _ = model.forward_encoder(x, self.mask_ratio, mask_2d=False)
        pred, _, _ = model.forward_decoder(latent, restore)
        # Same normalized masked patch reconstruction as official forward(),
        # avoiding its unrelated hard-coded .cuda() zero contrastive loss.
        return model.forward_loss(x, pred, mask, norm_pix_loss=True)

    def export_encoder(self):
        return {k: v for k, v in self.encoder.state_dict().items() if not k.startswith(('decoder_', 'mask_token', 'norm.'))}


class BEATsEncoder(nn.Module):
    def __init__(self, cfg, initialize=True):
        super().__init__()
        checkpoint = load_file(cfg['weights'])
        with official('unilm', 'beats') as source:
            module = source('BEATs')
            config = module.BEATsConfig(checkpoint['cfg'])
            if config.finetuned_model:
                raise ValueError('BEATs Iter3 requires BEATs_iter3 pure PT, not AudioSet-finetuned weights')
            self.encoder = module.BEATs(config)
        if initialize:
            strict_load(self.encoder, checkpoint['model'])
        self.output_dim = config.encoder_embed_dim
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=10)

    def prepare(self, clip):
        return dict(source=clip['audio'][None], padding_mask=torch.zeros(1, len(clip['audio']), dtype=torch.bool))

    def forward(self, batch):
        values, mask = self.encoder.extract_features(**batch)
        if mask is None:
            return values.mean(1)
        return (values * (~mask)[..., None]).sum(1) / (~mask).sum(1, keepdim=True).clamp_min(1)


class WavJEPA(nn.Module):
    """Bypass the upstream inference_mode API; use its actual trainable modules."""
    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        import importlib
        import sys
        import types
        source_path = str(path(cfg['weights'])) if cfg.get('weights') else cfg['hub_id']
        if not cfg.get('weights'):
            from huggingface_hub import snapshot_download
            source_path = snapshot_download(cfg['hub_id'], revision=cfg['revision'])
        # Transformers 4.57's remote-code cache misses transitive sibling files
        # for this release. A normal isolated Python package imports the exact
        # same pinned files without rewriting the model or cache internals.
        package_name = '_favor_wavjepa_' + cfg['revision']
        package = types.ModuleType(package_name)
        package.__path__ = [source_path]
        sys.modules[package_name] = package
        config_cls = importlib.import_module(package_name+'.configuration_wavjepa').WavJEPAConfig
        processor_cls = importlib.import_module(package_name+'.feature_extraction_wavjepa').WavJEPAFeatureExtractor
        model_cls = importlib.import_module(package_name+'.modeling_wavjepa').WavJEPAModel
        config = config_cls.from_pretrained(source_path)
        self.processor = processor_cls.from_pretrained(source_path)
        hf_model = hf_load(model_cls, cfg) if initialize else model_cls(config)
        self.pretraining = pretrain
        if pretrain:
            import copy
            with official('WavJEPA') as source:
                extractor = source('wavjepa.extractors.audio_feature_extractor').ConvFeatureExtractor(
                    conv_layers_spec=eval(config.extractor_config['conv_layers_spec'], {'__builtins__': {}}),
                    **{k: v for k, v in config.extractor_config.items() if k != 'conv_layers_spec'})
                self.encoder = source('wavjepa.jepa').JEPA(feature_extractor=extractor,
                    transformer_encoder_layers_cfg=copy.deepcopy(config.encoder_layers_cfg),
                    transformer_encoder_cfg=copy.deepcopy(config.encoder_cfg),
                    transformer_decoder_layers_cfg=copy.deepcopy(config.decoder_layers_cfg),
                    transformer_decoder_cfg=copy.deepcopy(config.decoder_cfg), process_audio_seconds=2.01,
                    average_top_k_layers=12, use_gradient_checkpointing=False)
                self.masker = source('wavjepa.masking').TimeInverseBlockMasker(
                    target_masks_per_context=4, context_mask_prob=.65, context_mask_length=10,
                    target_prob=.25, target_length=10, ratio_cutoff=.1, channel_based_masking=False)
            # The fixed HF release uses GELU. Current training-source defaults
            # use ReLU: keep the checkpoint's activation when continuing PT.
            for module in self.encoder.modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    module.activation = F.gelu
                    module.activation_relu_or_gelu = 2
            strict_load(self.encoder, hf_model.model.state_dict())
        else:
            self.encoder = hf_model.model
            for name in ('teacher_encoder', 'decoder', 'decoder_to_encoder_mapper',
                         'encoder_to_decoder_mapper', 'mask_token', 'pos_encoding_decoder'):
                delattr(self.encoder, name)
        self.output_dim = 768
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=2.01)

    def prepare(self, clip):
        # Native HF extractor applies RMS normalization, followed by native
        # per-segment standardization; fit neither operation across subjects.
        audio = self.processor([clip['audio'].numpy()], return_tensors='pt')['input_values']
        audio = torch.as_tensor(audio).reshape(1, 1, -1)
        length = audio.shape[-1]
        audio = F.pad(audio, (0, 32160-length))
        audio = (audio-audio.mean((-2, -1), keepdim=True)) / (audio.std((-2, -1), keepdim=True)+1e-5)
        real_tokens = min(200, max(1, self.encoder.extract_audio.total_patches(length)))
        mask = torch.arange(200)[None] >= real_tokens
        return dict(audio=audio, padding_mask=mask)

    def forward(self, batch):
        model = self.encoder
        x = model.feature_norms(model.extract_audio(batch['audio']))
        if model.post_extraction_mapper is not None:
            x = model.post_extraction_mapper(x)
        x = model.encoder(x+model.pos_encoding_encoder, src_key_padding_mask=batch['padding_mask'])
        valid = (~batch['padding_mask'])[..., None]
        return (x*valid).sum(1) / valid.sum(1).clamp_min(1)

    def pretrain_loss(self, batches, step=0):
        # The official WebAudio loader zero-pads short recordings. SSL keeps
        # its fixed-length native target; downstream readout uses real lengths.
        audio = torch.cat([b['audio'] for b in batches])
        context, targets, combined = self.masker(len(batches), self.encoder.total_patches, 1)
        return self.encoder(audio, context.to(audio.device), targets.to(audio.device), combined.to(audio.device))['loss']

    @torch.no_grad()
    def after_update(self, step):
        hp = self.encoder.hparams
        decay = hp.ema_end_decay - (hp.ema_end_decay-hp.ema_decay) * max(0., 1-step/max(1, hp.ema_anneal_end_step))
        for student, teacher in zip(self.encoder.encoder.parameters(), self.encoder.teacher_encoder.parameters()):
            teacher.mul_(decay).add_(student, alpha=1-decay)

    def export_encoder(self):
        prefixes = ('teacher_encoder.', 'decoder.', 'decoder_to_encoder_mapper.',
                    'encoder_to_decoder_mapper.', 'mask_token', 'pos_encoding_decoder')
        return {k: v for k, v in self.encoder.state_dict().items() if not k.startswith(prefixes)}


class EAT(nn.Module):
    """Use the official Fairseq release for both FT and continued SSL."""
    def __init__(self, cfg, initialize=True, pretrain=False):
        super().__init__()
        # Fairseq's official loader enforces strict=True, including EMA state.
        # It also restores the actual model/task dataclasses from this PT file.
        import fairseq
        with official('EAT') as source:
            source('models.EAT_pretraining')
            source('tasks')
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [str(path(cfg['weights']))], strict=True)
            self.encoder = models[0]
        logging.getLogger('baselines').info('EAT Fairseq strict checkpoint load: %s', cfg['weights'])
        self.pretraining = pretrain
        if not pretrain:
            self.encoder.remove_pretraining_modules()
        self.output_dim = self.encoder.cfg.embed_dim
        if self.output_dim != 1024:
            raise ValueError('EAT Large and continued EAT pretraining require the 20-epoch Large PT checkpoint')
        self.input_spec = dict(modality='audio', sample_rate=16000, seconds=10.24)

    def prepare(self, clip):
        x, valid = fbank(clip['audio'], mean=-4.268, std=2*4.569)
        return dict(features=x[None, None], valid=torch.tensor([valid]))

    def forward(self, batch):
        result = self.encoder.extract_features(batch['features'], mask=False, remove_extra_tokens=True)
        x = result['x']
        time = torch.arange(64, device=x.device).repeat_interleave(8)
        mask = time[None] < (batch['valid'][:, None]+15)//16
        if x.shape[1] != mask.shape[1]:
            raise ValueError('EAT token geometry differs from the fixed 1024x128 release')
        return (x*mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def pretrain_loss(self, batches, step=0):
        # Validation must not advance this Python counter: otherwise the next
        # training call would skip the native EMA update for that step.
        if self.training:
            self.encoder.set_num_updates(step)
        x = torch.cat([b['features'] for b in batches])
        result = self.encoder(x, mask=True, features_only=False)
        # Exactly Fairseq's model criterion: sum every enabled native loss,
        # divide by sample_size (masked patches), preserving checkpoint weights.
        return sum(value.float().sum() for value in result['losses'].values()) / result['sample_size']

    def get_extra_state(self):
        # Fairseq includes teacher tensors in _ema, but not this update counter.
        return {'num_updates': self.encoder.num_updates} if self.pretraining else {}

    def set_extra_state(self, state):
        if self.pretraining:
            self.encoder.num_updates = state['num_updates']

    def export_encoder(self):
        import copy
        # remove_pretraining_modules mutates both cfg and EMA; prune a copy so
        # exporting best.pt never disrupts the continuing training model.
        encoder = copy.deepcopy(self.encoder)
        encoder.remove_pretraining_modules()
        return encoder.state_dict()

    def counted_components(self):
        # Fairseq's EMA wrapper is not registered as a child nn.Module.
        return {'ema_teacher_outside_module_tree': self.encoder.ema.model} if self.pretraining else {}
