#!/usr/bin/env python3
"""Encode each fixed 20%-sampled audio-video record once per checkpoint."""
import argparse,json,os,sys,time,tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import yaml
import main
from datasets.va_lejepa.dataset import VALeJEPADataset
from datasets.va_lejepa.collate import collate_views,to_device

HERE=Path(__file__).resolve().parent.parent; BASE=HERE/'cache/sample20'; FEATURE=BASE/'features';FEATURE.mkdir(parents=True,exist_ok=True)

class Indexed(Dataset):
    def __init__(self,ds):self.ds=ds
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):return i,self.ds[i]

def collate(batch):
    good=[(i,s['global']) for i,s in batch if 'decode_error' not in s]
    bad=[(i,s['decode_error'].get('reason','decode error')) for i,s in batch if 'decode_error' in s]
    return ([i for i,_ in good],collate_views([v for _,v in good]) if good else None,bad)

def output_path(dataset,step):return FEATURE/f'{dataset}-{step}.npz'

def main_run():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--checkpoints',nargs='+',default=['step-10000','step-30000','step-50000','step-70000','step-80000']);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=4);p.add_argument('--config',default=str(HERE/'cache/early-stop/audio-video.yaml'));args=p.parse_args()
    os.chdir(HERE.parent);torch.set_num_threads(1);torch.manual_seed(42)
    cfg=yaml.safe_load(Path(args.config).read_text());items={c['id']:c for c in cfg['checkpoints']};summ=pd.read_csv(BASE/'sampling.csv').set_index('dataset');device=torch.device(f'cuda:{args.gpu}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    for step in args.checkpoints:
        if step not in items:raise ValueError('unknown checkpoint '+step)
        ck=items[step]; pending=[]
        for name in args.datasets:
            path=output_path(name,step)
            if path.is_file():
                with np.load(path,allow_pickle=False) as data:
                    if str(data['sample_sha256'])!=summ.loc[name,'sample_sha256'] or len(data['features'])!=int(summ.loc[name,'sampled_rows']):raise ValueError('incompatible existing feature cache '+str(path))
                print('reuse',name,step,flush=True)
            else:pending.append(name)
        if not pending:continue
        start=time.perf_counter();model,family,metadata=main.build_model(Path(ck['path']),'audio-video',cfg,device);model.eval();print('loaded',step,'seconds',round(time.perf_counter()-start,1),flush=True)
        for name in pending:
            d=pd.read_csv(BASE/(name+'.csv'),dtype=str,low_memory=False).fillna('');root=summ.loc[name,'media_root'];digest=summ.loc[name,'sample_sha256']
            settings=dict(datasets=[str(BASE/(name+'.csv'))],root=root,sample_rate=16000,local_views=0,global_seconds=6,local_seconds=2,video_fps=8,video_frames_global=48,video_frames_local=16,video_global_size=112,short_policy='zero_pad',minimum_seconds=.1,normalize_audio=True,on_decode_error='skip',sync=dict(strict=True,require_same_source_id=True,require_same_start=True,require_same_duration=True,duration_tolerance_seconds=.5,duration_tolerance_frames=2,offset_tolerance_seconds=.1,on_mismatch='error'))
            start=time.perf_counter();ds=VALeJEPADataset(settings,training=False)
            if len(ds)!=len(d):raise ValueError(f'{name}: manifest rows changed {len(ds)}/{len(d)}')
            dl=DataLoader(Indexed(ds),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate,pin_memory=False,multiprocessing_context='spawn' if args.workers else None,persistent_workers=args.workers>0)
            out=None;errors=[];seen=0
            for indices,view,bad in dl:
                errors.extend(bad)
                if indices:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        z=model.encode_view(to_device(view,device))[cfg['va_branch']].float().cpu().numpy()
                    if z.ndim!=2 or len(z)!=len(indices) or not np.isfinite(z).all():raise ValueError(f'{name}: invalid feature output')
                    if out is None:out=np.full((len(d),z.shape[1]),np.nan,dtype=np.float32)
                    out[indices]=z
                seen+=len(indices)+len(bad)
                if seen%max(args.batch_size*100,1)==0 or seen==len(d):print('progress',step,name,seen,'/',len(d),'bad',len(errors),'secs',round(time.perf_counter()-start,1),flush=True)
            if out is None:raise ValueError(f'{name}: no decoded features')
            valid=np.isfinite(out).all(axis=1);duration=time.perf_counter()-start
            with tempfile.NamedTemporaryFile(dir=FEATURE,suffix='.tmp',delete=False) as handle:
                tmp=Path(handle.name);np.savez_compressed(handle,features=out,valid=valid,sample_sha256=np.array(digest),feature_seconds=np.array(duration),checkpoint=np.array(step),batch_size=np.array(args.batch_size))
            os.replace(tmp,output_path(name,step))
            if errors:
                (FEATURE/f'{name}-{step}.errors.json').write_text(json.dumps({'decode_failures':len(errors),'examples':[(int(i),reason[:250]) for i,reason in errors[:50]]},ensure_ascii=False,indent=2)+'\n')
            print('finished',step,name,'valid',int(valid.sum()),'failed',int((~valid).sum()),'seconds',round(duration,1),flush=True)
            del dl,ds,out
        del model;torch.cuda.empty_cache()
if __name__=='__main__':main_run()
