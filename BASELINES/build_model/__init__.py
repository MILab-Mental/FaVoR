"""Only implemented adapters are in the factory; availability lives in README."""
import torch
from torch import nn

from BASELINES.common.runtime import load_file, strict_load


class TaskModel(nn.Module):
    def __init__(self, encoder, outputs, instance_cfg=None, encoder_batch_size=8):
        super().__init__()
        self.encoder = encoder
        self.encoder_batch_size = int(encoder_batch_size)
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

    def predict_sample(self,sample,task,device):
        return self.predict_samples([sample],task,device)[0]

    def predict_samples(self,samples,task,device):
        from collections import defaultdict
        from BASELINES.common.runtime import move,input_summary
        from utils.instance_evaluation import average_window_outputs
        requests,layouts,prepared_cache,encoded_cache=[],[],{},{}
        frames=getattr(self.encoder,'kind',None) in ('dinov2','dinov3','clip')
        native_batch=getattr(self.encoder,'batch_inputs',None)
        for sample_index,sample in enumerate(samples):
            sentences=[]
            for clips in sample['sentences']:
                indices=[]
                for clip_index,clip in enumerate(clips):
                    key=(sample_index,id(clips),clip_index)
                    if key not in prepared_cache:prepared_cache[key]=move(self.prepare(clip),device)
                    if self.training or key not in encoded_cache:
                        index=len(requests)
                        requests.append((clip,prepared_cache[key]))
                        encoded_cache[key]=index
                    indices.append(encoded_cache[key])
                sentences.append(indices)
            layouts.append(sentences)
        groups=defaultdict(list)
        for i,(_,prepared) in enumerate(requests):
            key=() if native_batch else tuple((k,tuple(v.shape if frames else v.shape[1:]),v.dtype) for k,v in prepared.items())
            groups[key].append(i)
        features=[None]*len(requests)
        for indices in groups.values():
            for start in range(0,len(indices),self.encoder_batch_size):
                selected=indices[start:start+self.encoder_batch_size]
                inputs=native_batch([requests[i][1] for i in selected]) if native_batch else {
                    k:(torch.stack if frames else torch.cat)([requests[i][1][k] for i in selected],dim=0)
                    for k in requests[selected[0]][1]}
                values=self.encoder(inputs)
                if values.shape[0]!=len(selected):raise ValueError('Encoder batch/readout axis mismatch')
                if not hasattr(self,'_io_check'):self._io_check=input_summary(requests[selected[0]][0],inputs,values)
                for i,value in zip(selected,values.split(1)):features[i]=value
        outputs=[]
        for sample,layout in zip(samples,layouts):
            sentences=[torch.stack([features[i] for i in indices]).mean(0) if self.instance_level
                       else average_window_outputs([self.head(features[i]) for i in indices],task) for indices in layout]
            windows=[self.head(torch.stack([sentences[i] for i in indices],dim=1)) if self.instance_level
                     else average_window_outputs([sentences[i] for i in indices],task) for indices in sample['windows']]
            output=average_window_outputs(windows,task)
            self._io_check['output_shape']=list(output.shape)
            if self.instance_level and 'instance' not in self._io_check:
                self._io_check['instance']=dict(sentences=len(sentences),unique_sentences=len({id(x) for x in sample['sentences']}),
                    windows=len(windows),window_sentences=len(sample['windows'][0]))
            outputs.append(output)
        return outputs

    def gradient_components(self):
        return self.encoder.gradient_components() if hasattr(self.encoder, 'gradient_components') else {}

    def counted_components(self):
        return self.encoder.counted_components() if hasattr(self.encoder,'counted_components') else {}


def build_model(cfg, *, pretrain=False, initialize=True):
    if initialize and cfg['model'].get('weight_sha256'):
        from BASELINES.common.runtime import fingerprint
        if fingerprint(cfg['model']['weights'])!=cfg['model']['weight_sha256']:
            raise ValueError('Initialization checkpoint SHA256 differs from the pinned download')
    from .audio import WaveformEncoder, WhisperEncoder, AudioMAE, BEATsEncoder, WavJEPA, EAT
    from .video import FrameEncoder, VideoMAE, MAEDFER, VJEPA21, HieraVideo, MaskFeat
    from .audiovisual import CAVMAE, SocialMAE, LeAVJEPA, LanguageBind
    from .avhubert import AVHubert
    from .avsiam import AVSiam,AVSiamLarge
    factories = dict(waveform=WaveformEncoder, whisper=WhisperEncoder,
        audiomae=AudioMAE, beats=BEATsEncoder, dinov2=FrameEncoder,
        dinov3=FrameEncoder, clip=FrameEncoder, videomae=VideoMAE,
        maedfer=MAEDFER, vjepa21=VJEPA21, hiera=HieraVideo,
        cavmae=CAVMAE, socialmae=SocialMAE, leavjepa=LeAVJEPA, wavjepa=WavJEPA,
        eat=EAT, maskfeat=MaskFeat, avhubert=AVHubert, languagebind=LanguageBind, avsiam=AVSiam,
        avsiam_large=AVSiamLarge)
    model_cfg = cfg['model']
    kind = model_cfg['kind']
    import yaml
    from BASELINES.common.official import LOCK
    source_names = dict(audiomae='AudioMAE', videomae='VideoMAE', maedfer='MAE-DFER',
        dinov2='dinov2', vjepa21='vjepa2', hiera='hiera', beats='unilm',
        cavmae='cav-mae', socialmae='SocialMAE', leavjepa='LeAVJEPA', wavjepa='WavJEPA',
        eat='EAT', maskfeat='SlowFast', avhubert='av_hubert', languagebind='LanguageBind', avsiam='AVSiam',avsiam_large='AVSiam')
    if kind in source_names:
        name = source_names[kind]
        cfg['upstream_code'] = {name: yaml.safe_load(LOCK.read_text())[name]}
        if kind in ('eat','avhubert'):
            fairseq_source='fairseq-avhubert' if kind=='avhubert' else 'fairseq'
            cfg['upstream_code'][fairseq_source]=yaml.safe_load(LOCK.read_text())[fairseq_source]
    if kind not in factories:
        raise ValueError(f'No implemented adapter for {kind}; see README support table')
    transfer = model_cfg.get('encoder_checkpoint')
    route = cfg.get('route', 'direct_finetune')
    if kind=='socialmae' and not pretrain and not transfer:
        raise ValueError('Social-MAE requires the research-pretrained encoder; direct fine-tuning is not an experiment')
    if pretrain:
        if kind not in ('audiomae', 'videomae', 'cavmae', 'leavjepa', 'wavjepa', 'eat', 'maskfeat','avsiam_large','socialmae'):
            raise ValueError(f'No native pretraining implementation for {kind}')
        if route not in ('scratch_pretrain', 'continue_pretrain'):
            raise ValueError('Pretraining route must be explicit')
        if route == 'scratch_pretrain' and kind not in ('audiomae', 'leavjepa','avsiam_large','socialmae'):
            raise ValueError('This experiment requires the specified public initialization')
        if kind == 'audiomae' and model_cfg.get('size') != 'large':
            raise ValueError('AudioMAE scratch pretraining uses the random-initialized Large architecture')
        if route == 'continue_pretrain' and kind in ('audiomae', 'leavjepa','avsiam_large','socialmae'):
            raise ValueError('This table entry is a scratch-pretraining experiment')
    kwargs = dict(initialize=initialize and not transfer and route != 'scratch_pretrain')
    if kind in ('audiomae', 'videomae', 'cavmae', 'leavjepa', 'wavjepa', 'eat', 'maskfeat','avsiam_large','socialmae'):
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
    return TaskModel(model, outputs, model_cfg.get('instance_aggregator'),model_cfg.get('encoder_batch_size',8))
