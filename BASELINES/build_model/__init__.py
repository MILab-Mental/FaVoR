"""Only implemented adapters are in the factory; availability lives in README."""
import torch
from torch import nn

from BASELINES.common.runtime import load_file, strict_load


class TaskModel(nn.Module):
    def __init__(self, encoder, outputs, instance_cfg=None):
        super().__init__()
        self.encoder = encoder
        self.instance_level = instance_cfg is not None
        if self.instance_level:
            from models.instance_aggregator import InstancePredictionHead
            self.head = InstancePredictionHead(encoder.output_dim, outputs, instance_cfg)
        else:
            self.head = nn.Linear(encoder.output_dim, outputs)
        self.input_spec = encoder.input_spec

    def prepare(self, clip):
        return self.encoder.prepare(clip)

    def forward(self, batch):
        features = self.encoder(batch)
        return self.head(features[:, None] if self.instance_level else features)

    def predict_sample(self, sample, task, device):
        from BASELINES.common.runtime import move, input_summary
        from utils.instance_evaluation import average_window_outputs
        sentences, encoded, views = [], {}, {}
        for clips in sample['sentences']:
            key = id(clips)
            # Repeat the same augmented inputs for short instances, while
            # retaining native training dropout in every sentence slot.
            if self.training or key not in encoded:
                if key not in views:
                    views[key] = [move(self.prepare(clip), device) for clip in clips]
                values = []
                for clip, prepared in zip(clips, views[key]):
                    value = self.encoder(prepared) if self.instance_level else self(prepared)
                    if not hasattr(self, '_io_check'):
                        self._io_check = input_summary(clip, prepared, value)
                    values.append(value)
                encoded[key] = torch.stack(values).mean(0) if self.instance_level else average_window_outputs(values, task)
                if not self.training:
                    del views[key]
            sentences.append(encoded[key])
        windows = [self.head(torch.stack([sentences[i] for i in indices], dim=1))
                   if self.instance_level else average_window_outputs([sentences[i] for i in indices], task)
                   for indices in sample['windows']]
        output = average_window_outputs(windows, task)
        self._io_check['output_shape'] = list(output.shape)
        if self.instance_level and 'instance' not in self._io_check:
            self._io_check['instance'] = dict(sentences=len(sentences),
                unique_sentences=len(encoded), windows=len(windows),
                window_sentences=len(sample['windows'][0]))
        return output

    def gradient_components(self):
        return self.encoder.gradient_components() if hasattr(self.encoder, 'gradient_components') else {}


def build_model(cfg, *, pretrain=False, initialize=True):
    from .audio import WaveformEncoder, WhisperEncoder, AudioMAE, BEATsEncoder, WavJEPA, EAT
    from .video import FrameEncoder, VideoMAE, MAEDFER, VJEPA21, HieraVideo, MaskFeat
    from .audiovisual import SocialMAE, CAVMAE, LeAVJEPA, AVHubert, LanguageBind, FavorNoJoint, AVSiam
    factories = dict(waveform=WaveformEncoder, whisper=WhisperEncoder,
        audiomae=AudioMAE, beats=BEATsEncoder, dinov2=FrameEncoder,
        dinov3=FrameEncoder, clip=FrameEncoder, videomae=VideoMAE,
        maedfer=MAEDFER, vjepa21=VJEPA21, hiera=HieraVideo,
        socialmae=SocialMAE, cavmae=CAVMAE, leavjepa=LeAVJEPA, wavjepa=WavJEPA,
        eat=EAT, maskfeat=MaskFeat, avhubert=AVHubert, languagebind=LanguageBind, avsiam=AVSiam)
    model_cfg = cfg['model']
    kind = model_cfg['kind']
    import yaml
    from BASELINES.common.official import LOCK
    source_names = dict(audiomae='AudioMAE', videomae='VideoMAE', maedfer='MAE-DFER',
        dinov2='dinov2', vjepa21='vjepa2', hiera='hiera', beats='unilm',
        socialmae='SocialMAE', cavmae='cav-mae', leavjepa='LeAVJEPA', wavjepa='WavJEPA',
        eat='EAT', maskfeat='SlowFast', avhubert='av_hubert', languagebind='LanguageBind', avsiam='AVSiam')
    if kind in source_names:
        name = source_names[kind]
        cfg['upstream_code'] = {name: yaml.safe_load(LOCK.read_text())[name]}
    if kind == 'favor_no_joint':
        if pretrain:
            raise ValueError('The FaVoR no-joint-pretraining control skips joint SSL')
        return FavorNoJoint(cfg, initialize=initialize)
    if kind not in factories:
        raise ValueError(f'No implemented adapter for {kind}; see README support table')
    transfer = model_cfg.get('encoder_checkpoint')
    route = cfg.get('route', 'direct_finetune')
    if pretrain:
        if kind not in ('audiomae', 'videomae', 'cavmae', 'leavjepa', 'wavjepa', 'eat', 'maskfeat'):
            raise ValueError(f'No native pretraining implementation for {kind}')
        if route not in ('scratch_pretrain', 'continue_pretrain'):
            raise ValueError('Pretraining route must be explicit')
        if route == 'scratch_pretrain' and kind not in ('audiomae', 'leavjepa'):
            raise ValueError('This experiment requires the specified public initialization')
        if kind == 'audiomae' and model_cfg.get('size') != 'large':
            raise ValueError('AudioMAE scratch pretraining uses the random-initialized Large architecture')
        if route == 'continue_pretrain' and kind in ('audiomae', 'leavjepa'):
            raise ValueError('This table entry is a scratch-pretraining experiment')
    kwargs = dict(initialize=initialize and not transfer and route != 'scratch_pretrain')
    if kind in ('audiomae', 'videomae', 'cavmae', 'leavjepa', 'wavjepa', 'eat', 'maskfeat'):
        kwargs['pretrain'] = pretrain
    model = factories[kind](model_cfg, **kwargs)
    if transfer and initialize:
        checkpoint = load_file(transfer)
        source = checkpoint['config']['model']
        if source['kind'] != kind or source.get('size') != model_cfg.get('size'):
            raise ValueError('Pretrained encoder architecture/size differs from finetuning model')
        allowed = ('fc_norm.',) if kind in ('audiomae', 'videomae') else ()
        if hasattr(model, 'load_encoder'):
            model.load_encoder(checkpoint['encoder'])
        else:
            strict_load(model.encoder, checkpoint['encoder'], allow_missing=allowed)
    if pretrain:
        return model
    from BASELINES.common.data import configure_instances
    configure_instances(cfg, model.input_spec['modality'])
    outputs = 1 if cfg['data']['task'] == 'regression' else int(cfg['data']['num_class'])
    return TaskModel(model, outputs, model_cfg.get('instance_aggregator'))
