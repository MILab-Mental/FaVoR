import copy
import csv
import itertools
import random
from pathlib import Path
import pytest
import torch
from torch import nn
from models.video_jepa.vision_transformer import VisionTransformer
from models.audio_lejepa import AudioLeEncoder
from models.va_lejepa import VALeJEPA, VABranchLoss, load_video_encoder_init, load_audio_encoder_init, save_checkpoint, load_checkpoint, cnn_geometry
from models.va_lejepa.temporal_alignment import video_bin_edges, align_audio_to_video
from models.va_lejepa.temporal_types import TemporalTokenOutput
from models.va_lejepa.encoders import video_token_output
from datasets.va_lejepa.va_manifest import read_manifest
from datasets.va_lejepa.manifest_tools import convert
from datasets.va_lejepa.av_decoder import validate_sync
from datasets.va_lejepa.av_time_sampler import AVTimeSampler, frame_indices
from datasets.va_lejepa import VALeJEPADataset, collate_va_lejepa
from app.pretrain_va_lejepa.train import _optimizer, _schedule
from models.video_lejepa import ModelEMA


def tiny_audio():
    return {'audio':{'sample_rate':100,'process_seconds':2,'max_process_seconds':2},
        'backbone':{'mode':'emotion2vec','conv_pos_depth':1,'conv_pos_width':3,'conv_pos_groups':2,'prenet_depth':1,'num_extra_tokens':2},
        'extractor':{'conv_layers':[[8,4,2],[8,2,2]],'mode':'layer_norm'},
        'encoder':{'depth':1,'d_model':8,'nhead':2,'dim_feedforward':16,'dropout':0.,'attention_dropout':0.,'ffn_targets':True}}


def tiny_va(modes=('feature',)):
    video = VisionTransformer(img_size=16,patch_size=8,num_frames=4,tubelet_size=1,embed_dim=24,depth=1,num_heads=3,
                              use_rope=True,use_cls_token=True,token_drop_rate=0,attn_mode='block_causal')
    cfg = dict(enabled=list(modes),common_dim=12,projection_dim=6,projector_hidden_dim=16,input_adapter={'hidden_dim':16})
    for mode in modes:
        cfg[mode] = dict(depth=1,nhead=3,max_video_tokens_per_time=2,max_audio_tokens_per_time=2,gradient_checkpointing=True)
    return VALeJEPA(video,AudioLeEncoder(tiny_audio()),cfg,100,True)


def view(batch=2,frames=4,seconds=2.,start=0.):
    return dict(video=torch.randn(batch,3,frames,16,16),audio=torch.randn(batch,round(seconds*100)),
        audio_lengths=torch.full((batch,),round(seconds*100)),start_time=torch.full((batch,),start,dtype=torch.float64),
        end_time=torch.full((batch,),start+seconds,dtype=torch.float64),
        video_frame_times=(start+torch.arange(frames,dtype=torch.float64)*seconds/frames)[None].expand(batch,-1),
        video_frame_indices=torch.arange(frames)[None].expand(batch,-1),
        audio_sample_range=torch.tensor([round(start*100),round((start+seconds)*100)])[None].expand(batch,-1))


def batch_data(batch=2,locals_=2):
    global_ = view(batch)
    locals_list = [view(batch,2,1.,.25*k) for k in range(locals_)]
    return {'global':global_,'local':{k:torch.stack([v[k] for v in locals_list],1) for k in locals_list[0]}}


@pytest.mark.parametrize('modes',[combo for n in range(1,4) for combo in itertools.combinations(['early','feature','late'],n)])
def test_all_fusion_combinations_share_encoders_backward(modes):
    model = tiny_va(modes)
    counts={'video':0,'audio':0}
    def count(name):
        def hook(*args): counts[name]+=1
        return hook
    model.video_encoder.register_forward_hook(count('video'))
    model.audio_encoder.register_forward_hook(count('audio'))
    output=model(batch_data())
    assert set(output)==set(modes)
    assert counts=={'video':3,'audio':3}
    for value in output.values():
        assert value['global'].shape==(2,1,6) and value['local'].shape==(2,2,6)
    loss=VABranchLoss(model,{'sigreg':{'knots':5,'num_proj':8}})(output)['loss']
    loss.backward()
    for mode in modes:
        assert model.input_adapters[mode].video[0].weight.grad is not None
        assert model.input_adapters[mode].audio[0].weight.grad is not None
    assert model.video_encoder.patch_embed.proj.weight.grad.norm()>0
    assert model.audio_encoder.backbone.feature_extractor.conv_blocks[0].conv.weight.grad.norm()>0


def test_cnn_and_physical_bins_empty_padding():
    extractor=AudioLeEncoder(tiny_audio()).backbone.feature_extractor
    assert cnn_geometry(extractor)==(4,6,2.5)
    times=torch.tensor([[1.,1.2,1.6]],dtype=torch.float64)
    edges=video_bin_edges(times,torch.tensor([1.]),torch.tensor([2.]))
    assert torch.allclose(edges,torch.tensor([[1.,1.1,1.4,2.]],dtype=torch.float64))
    video=TemporalTokenOutput(torch.zeros(1,3,1,2),times,torch.zeros(1,3,dtype=torch.bool),torch.zeros(1,2),3,1)
    audio=TemporalTokenOutput(torch.tensor([[[2.,4.],[8.,10.],[100.,100.],[50.,50.]]]),
        torch.tensor([[1.05,1.1,1.5,2.]],dtype=torch.float64),torch.tensor([[False,False,True,False]]),torch.zeros(1,2),4)
    aligned=align_audio_to_video(video,audio,torch.tensor([1.]),torch.tensor([2.]))
    assert aligned.assignment.tolist()==[[0,1,-1,-1]]
    assert aligned.audio_counts.tolist()==[[1,1,0]]
    assert aligned.audio_padding_mask.tolist()==[[False,False,True]]
    assert aligned.audio_features[0,0].tolist()==[2.,4.]


def test_sequence_order_budget_and_mask_invariance():
    model=tiny_va(('early','feature','late')).eval()
    data=view()
    reps,aligned=model.encode_view(data,True)
    aligned.audio.padding_mask[:,1:]=True
    aligned.assignment[:,1:]=-1
    aligned.audio_features.zero_()
    aligned.audio_padding_mask[:]=True
    seq,mask=model.fusion_adapters['feature'].build_sequence(aligned,model.input_adapters['feature'])
    assert seq.shape==(2,9,12) and mask[:,2::2].all()
    seq,_=model.fusion_adapters['late'].build_sequence(aligned,model.input_adapters['late'])
    assert seq.shape==(2,3,12)
    seq,_=model.fusion_adapters['early'].build_sequence(aligned,model.input_adapters['early'])
    assert seq.shape[1]<=1+4*(2+2)
    before=model.fusion_adapters['early'](aligned,model.input_adapters['early'])
    aligned.audio.tokens[:,1:]=1e9
    after=model.fusion_adapters['early'](aligned,model.input_adapters['early'])
    assert torch.allclose(before,after)


def test_sparse_video_rejected_and_dense_shape():
    model=tiny_va()
    output=video_token_output(model.video_encoder,view())
    assert output.tokens.shape==(2,4,4,24)
    model.video_encoder.token_drop_rate=.95
    with pytest.raises(ValueError,match='sparse'):
        video_token_output(model.video_encoder,view())


@pytest.mark.parametrize('modality',['video','audio'])
@pytest.mark.parametrize('weight_source',['encoder','ema'])
def test_initialization_schema_prefix_target_and_ema(tmp_path,modality,weight_source):
    model=tiny_va()
    target=model.video_encoder if modality=='video' else model.audio_encoder.backbone
    prefix='encoder.' if modality=='video' else 'encoder.backbone.'
    state=copy.deepcopy(target.state_dict())
    shadow={f'module.{prefix}{k}':v+1 for k,v in state.items() if v.is_floating_point()}
    path=tmp_path/'init.pt'
    torch.save({'schema':f'favor.{modality}_lejepa.v1','encoder':state,'state_dict_ema':shadow},path)
    cfg={'init_source':f'{modality}_lejepa','weight_source':weight_source,f'{modality}_lejepa_checkpoint':str(path)}
    loader=load_video_encoder_init if modality=='video' else load_audio_encoder_init
    report=loader(model.video_encoder if modality=='video' else model.audio_encoder,cfg)
    assert report['loaded_ratio']==1
    key=next(k for k,v in state.items() if v.is_floating_point())
    assert torch.equal(target.state_dict()[key],state[key]+(1 if weight_source=='ema' else 0))
    torch.save({'schema':'wrong','encoder':state},path)
    with pytest.raises(ValueError,match='expected'):
        loader(model.video_encoder if modality=='video' else model.audio_encoder,cfg)
    torch.save({'schema':f'favor.{modality}_lejepa.v1','encoder':state,'state_dict_ema':{'bad.'+k:v for k,v in state.items()}},path)
    if weight_source=='ema':
        with pytest.raises(ValueError,match='prefix'):
            loader(model.video_encoder if modality=='video' else model.audio_encoder,cfg)


def test_raw_loader_routing(monkeypatch):
    import models.va_lejepa.checkpoint as cp
    calls=[]
    monkeypatch.setattr(cp,'load_vjepa_encoder',lambda *args:(calls.append('v') or {}))
    monkeypatch.setattr(cp,'load_emotion2vec_encoder',lambda *args:(calls.append('a') or {}))
    model=tiny_va()
    cp.load_video_encoder_init(model.video_encoder,{'init_source':'vjepa','vjepa_checkpoint':'v.pt'})
    cp.load_audio_encoder_init(model.audio_encoder,{'init_source':'emotion2vec','emotion2vec_checkpoint':'a.pt'})
    assert calls==['v','a']


def test_manifest_convert_spaces_and_reject_identity(tmp_path):
    legacy=tmp_path/'legacy.txt'
    legacy.write_text('/a space/x.mp4 /b space/x.wav 0\n/a/y.mp4 /b/z.wav 1\n')
    canonical=tmp_path/'canonical.csv'
    assert convert(legacy,canonical,tmp_path/'reject.csv')==(1,1)
    rows=read_manifest(canonical)
    assert rows[0]['source_id']=='x' and rows[0]['video_path']=='/a space/x.mp4'
    canonical.write_text(canonical.read_text().replace('/b space/x.wav','/b space/y.wav'))
    with pytest.raises(ValueError,match='identity'):
        read_manifest(canonical)


@pytest.mark.parametrize('field,value',[('duration',6.1),('start',.1)])
def test_strict_sync_reject(field,value):
    video={'duration':6.,'start':0.,'sample_rate':0}
    audio={'duration':6.,'start':0.,'sample_rate':16000}
    audio[field]=value
    with pytest.raises(ValueError,match='mismatch'):
        validate_sync(video,audio,{'pair_id':'x'}, {})


def test_sampler_and_real_fps():
    random.seed(3)
    global_,local=AVTimeSampler()(10.,10.)
    assert global_[1]-global_[0]==pytest.approx(6.)
    assert all(global_[0]<=start<end<=global_[1] for start,end in local)
    indices,times=frame_indices(*global_,48,29.97,300)
    assert torch.equal(times,indices.double()/29.97)
    assert indices.unique().numel()==48
    with pytest.raises(ValueError,match='shorter'):
        AVTimeSampler(short_policy='error')(4.,4.)


def test_two_steps_checkpoint_resume_branch_expansion(tmp_path):
    model=tiny_va(('feature','late'))
    cfg=dict(lr_video_pretrained=1e-4,lr_audio_pretrained=2e-4,lr_new=1e-3,total_steps=3,warmup_steps=1,weight_decay=.01)
    optimizer=_optimizer(model,cfg)
    ema=ModelEMA(model,update_every=1)
    criterion=VABranchLoss(model,{'sigreg':{'num_proj':4,'knots':3}})
    for step in range(2):
        optimizer.zero_grad()
        loss=criterion(model(batch_data()))['loss']
        loss.backward()
        _schedule(optimizer,step,cfg)
        optimizer.step()
        ema.update(model,step+1)
    ratios={g['group_name']:g['lr']/g['base_lr'] for g in optimizer.param_groups}
    assert len(set(ratios.values()))==1
    path=tmp_path/'latest.pt'
    save_checkpoint(path,model,optimizer,ema=ema,step=2,epoch=1,config=cfg)
    clone=tiny_va(('feature','late'))
    checkpoint=load_checkpoint(path,clone,_optimizer(clone,cfg),ema=ModelEMA(clone))
    load_checkpoint(path,clone,scaler=torch.amp.GradScaler('cuda',enabled=False))
    assert checkpoint['step']==2 and 'global_step' not in checkpoint
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
    expanded=tiny_va(('early','feature','late'))
    load_checkpoint(path,expanded,allow_new_branches=True)
    ddp=dict(checkpoint,model={'module.'+k:v.clone() for k,v in checkpoint['model'].items()})
    torch.save(ddp,tmp_path/'ddp.pt')
    load_checkpoint(tmp_path/'ddp.pt',clone)


def test_dataset_pair_owner_and_decode_failure(tmp_path):
    manifest=tmp_path/'pairs.csv'
    manifest.write_text('pair_id,source_id,video_path,audio_path\nx,x,/a/x.mp4,/b/x.wav\ny,y,/a/y.mp4,/b/y.wav\n')
    class Decoder:
        def __init__(self,row,sample_rate,sync):
            self.video_duration=self.audio_duration=8.
            self.diagnostics={'duration_delta':0.,'start_delta':0.,'repeated':False,'padded':False}
        def decode(self,interval,frames):
            start,end=interval
            return dict(video=torch.zeros(frames,16,16,3,dtype=torch.uint8),audio=torch.ones(round((end-start)*100)),
                audio_lengths=torch.tensor(round((end-start)*100)),start_time=torch.tensor(start),end_time=torch.tensor(end),
                video_frame_times=start+torch.arange(frames)*(end-start)/frames,video_frame_indices=torch.arange(frames),
                audio_sample_range=torch.tensor([round(start*100),round(end*100)]))
    cfg=dict(manifests=[str(manifest)],sample_rate=100,global_seconds=6.,local_seconds=2.,local_views=4,
        video_frames_global=48,video_frames_local=16,video_fps=8,video_global_size=16,video_local_size=8)
    dataset=VALeJEPADataset(cfg,decoder_factory=Decoder)
    sample=dataset[0]
    assert sample['pair_id']=='x'
    batch=collate_va_lejepa([sample,dataset[1]])
    assert batch['global']['video'].shape==(2,3,48,16,16)
    assert batch['global']['audio'].shape==(2,600)
    assert batch['local']['audio'].shape==(2,4,200)
    class Failed(Decoder):
        def decode(self,*args): raise ValueError('audio broken')
    dataset.decoder_factory=Failed
    with pytest.raises(RuntimeError,match='pair_id=x'):
        dataset[0]


def test_pretrain_entry_accumulation_and_full_resume(tmp_path,monkeypatch):
    import app.pretrain_va_lejepa.train as train
    from torch.utils.data import DataLoader, Dataset
    class Samples(Dataset):
        epoch=0
        def __len__(self): return 3
        def __getitem__(self,index):
            sample=view(1)
            local=view(1,2,1.)
            return {'global':{k:v[0] for k,v in sample.items()},'local':[{k:v[0] for k,v in local.items()}],
                'pair_id':str(index),'source_id':str(index),'video_path':'v','audio_path':'a',
                'sync_diagnostics':{'duration_delta':0.,'start_delta':0.,'repeated':False,'padded':False}}
    class Sampler:
        def set_epoch(self,epoch): pass
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(train,'build_va_lejepa',lambda *args,**kwargs:tiny_va(('feature','late')))
    monkeypatch.setattr(train,'make_va_lejepa_loader',lambda *args:(None,DataLoader(Samples(),batch_size=1,collate_fn=collate_va_lejepa),Sampler()))
    cfg=dict(folder=str(tmp_path),model={},data={},optimization=dict(total_steps=2,warmup_steps=1,grad_accum_steps=2,
        lr_video_pretrained=1e-4,lr_audio_pretrained=2e-4,lr_new=1e-3),meta=dict(auto_resume=True,log_every_steps=1,save_every_steps=1),
        ema=dict(enabled=True,update_every=1),loss={'sigreg':{'num_proj':4,'knots':3}})
    train.main(cfg)
    cfg['optimization']['total_steps']=3
    train.main(cfg)
    state=torch.load(tmp_path/'latest.pt',weights_only=False)
    assert state['step']==3 and state['optimizer'] and state['ema']
    assert len(state['rng_states'])==1
    rows=list(csv.DictReader((tmp_path/'logs/history.csv').open()))
    assert [int(r['step']) for r in rows]==[1,2,3]


def test_branch_sigreg_remains_float32_under_autocast():
    model=tiny_va()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model(batch_data())
        losses=VABranchLoss(model,{'sigreg':{'knots':3,'num_proj':4}})(output)
    assert losses['branches']['feature']['sigreg_loss'].dtype==torch.float32
    assert losses['loss'].dtype==torch.float32


def test_equal_backbone_dimensions_still_use_independent_adapters():
    cfg=tiny_audio()
    cfg['encoder'].update(d_model=24,nhead=3,dim_feedforward=48)
    video=tiny_va().video_encoder
    model=VALeJEPA(video,AudioLeEncoder(cfg),dict(enabled=['feature'],common_dim=12,projection_dim=6,
        projector_hidden_dim=16,input_adapter={'hidden_dim':16},feature={'depth':1,'nhead':3}),100)
    adapters=model.input_adapters['feature']
    assert adapters.video[0].weight is not adapters.audio[0].weight
    loss=VABranchLoss(model,{'sigreg':{'num_proj':4,'knots':3}})(model(batch_data()))['loss']
    loss.backward()
    assert adapters.video[0].weight.grad.norm()>0 and adapters.audio[0].weight.grad.norm()>0


def test_actual_default_audio_cnn_geometry_and_branch_loss_weights():
    from app.main import load_config
    from models.audio_jepa.extractor import ConvFeatureExtractor
    cfg=load_config('CONFIGS/models/pretrain/audio-video/lejepa/vitl-emotion2vec.yaml')['model']['audio']['extractor']
    extractor=ConvFeatureExtractor(cfg['conv_layers'])
    assert cnn_geometry(extractor)==(320,400,199.5)
    assert extractor.output_length(6*16000)==299
    model=tiny_va(('feature','late'))
    model.loss_weights={'feature':.25,'late':2.}
    losses=VABranchLoss(model,{'sigreg':{'num_proj':4,'knots':3}})(model(batch_data()))
    assert torch.allclose(losses['loss'],.25*losses['branches']['feature']['loss']+2*losses['branches']['late']['loss'])


@pytest.mark.parametrize('seconds', [.1, 1., 4., 6., 8.])
def test_short_sampler_preserves_real_interval(seconds):
    global_, locals_ = AVTimeSampler(training=False)(seconds, seconds)
    assert global_[1] - global_[0] == pytest.approx(min(seconds, 6.))
    assert all(global_[0] <= a < b <= global_[1] for a, b in locals_)
    assert all(b - a == pytest.approx(min(seconds, 2.)) for a, b in locals_)
    with pytest.raises(ValueError, match='minimum_seconds'):
        AVTimeSampler()(0., 0.)


def test_short_dataset_zero_padding_after_normalization(tmp_path):
    manifest = tmp_path / 'short.csv'
    manifest.write_text('pair_id,source_id,video_path,audio_path\nx,x,/a/x.mp4,/b/x.wav\n')
    class Decoder:
        def __init__(self, *args):
            self.video_duration = self.audio_duration = 1.5
            self.diagnostics = {'padded': False, 'repeated': False}
        def decode(self, interval, frames):
            start, end = interval
            length = round((end - start) * 100)
            return dict(video=torch.ones(frames,16,16,3,dtype=torch.uint8), audio=torch.arange(length).float(),
                audio_lengths=torch.tensor(length), start_time=torch.tensor(start), end_time=torch.tensor(end),
                video_frame_times=start+torch.arange(frames)*(end-start)/frames,
                video_frame_indices=torch.arange(frames), audio_sample_range=torch.tensor([0,length]))
    cfg = dict(manifests=[str(manifest)],sample_rate=100,video_global_size=16,video_local_size=8)
    dataset = VALeJEPADataset(cfg, training=False, decoder_factory=Decoder)
    sample = dataset[0]
    for item, frames, samples in [(sample['global'],48,600), *[(v,16,200) for v in sample['local']]]:
        assert item['video'].shape == (3,frames,16,16)
        assert item['audio'].shape == (samples,)
        assert item['video_lengths'] == 12 and item['audio_lengths'] == 150
        assert item['end_time'] - item['start_time'] == 1.5
        assert torch.count_nonzero(item['video'][:,12:]) == 0
        assert torch.count_nonzero(item['audio'][150:]) == 0
        assert item['video_padding_mask'].sum() == frames - 12
        assert item['audio_padding_mask'].sum() == samples - 150
    assert sample['sync_diagnostics']['padded'] and not sample['sync_diagnostics']['repeated']
    batch = collate_va_lejepa([sample, sample])
    assert batch['local']['video_lengths'].shape == (2,4)
    dataset = VALeJEPADataset(dict(cfg,short_policy='error'), decoder_factory=Decoder)
    with pytest.raises(RuntimeError,match='shorter'):
        dataset[0]


def test_padded_mixed_lengths_encode_and_backward_ignore_padding():
    import torch.nn.functional as F
    model = tiny_va(('early','feature','late')).eval()
    long = view(1)
    short = view(1,2,1.)
    expected = model.encode_view(short)
    padded = dict(short)
    padded['video'] = F.pad(short['video'], (0,0,0,0,0,2))
    padded['audio'] = F.pad(short['audio'], (0,100))
    for key in ('video_frame_times','video_frame_indices'):
        padded[key] = F.pad(short[key], (0,2))
    mixed = {key: torch.cat([long[key], padded[key]], 0) for key in long}
    mixed['video_lengths'] = torch.tensor([4,2])
    output = model.encode_view(mixed)
    for mode in model.enabled_modes:
        assert torch.allclose(output[mode][1:], expected[mode], atol=1e-5)
    mixed['video'][1,:,2:] = 1e6
    mixed['audio'][1,100:] = 1e6
    after, grouped = model.encode_view(mixed, return_alignment=True)
    assert isinstance(grouped, list) and len(grouped) == 2
    for mode in model.enabled_modes:
        assert torch.allclose(output[mode], after[mode], atol=1e-5)
    sum(result.sum() for result in after.values()).backward()
    assert model.video_encoder.patch_embed.proj.weight.grad is not None
    assert model.audio_encoder.backbone.feature_extractor.conv_blocks[0].conv.weight.grad is not None



def test_real_short_media_decode_zero_padding(tmp_path):
    import shutil
    import subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    pytest.importorskip('decord')
    video = tmp_path / 'short.mp4'
    audio = tmp_path / 'short.wav'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=32x32:r=16:d=1.5',
                    '-an','-c:v','mpeg4',str(video)], check=True)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1.5',
                    str(audio)], check=True)
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text(f'pair_id,source_id,video_path,audio_path\nshort,short,{video},{audio}\n')
    dataset = VALeJEPADataset(dict(manifests=[str(manifest)],local_views=0,video_global_size=16),training=False)
    sample = dataset[0]
    view_ = sample['global']
    assert view_['video_lengths'] == 12 and view_['audio_lengths'] == 24000
    assert view_['audio'].shape == (96000,)
    assert torch.count_nonzero(view_['audio'][24000:]) == 0
    assert torch.count_nonzero(view_['video'][:,12:]) == 0
    assert view_['end_time'] == 1.5
    assert sample['sync_diagnostics']['padded']
