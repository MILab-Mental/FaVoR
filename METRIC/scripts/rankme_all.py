#!/usr/bin/env python3
"""Rank all available VA-LeJEPA checkpoints on the original 20% sample."""
import json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from score20 import labels

HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parent;BASE=HERE/'cache/sample20';OUT=HERE/'results/sample20';DETAILS=OUT/'details';LOG=HERE/'logs'
TRAJECTORY=Path('/home/data/sdd/mental/OUTPUTS/pretrain_va_lejepa/JEPA-e5-latest-C1')
I3=Path('/home/data/sdd/mental/OUTPUTS/pretrain_va_lejepa/I3-C1/latest.pt')
STEPS=[f'step-{n}' for n in range(5000,80001,5000)]+['I3-C1-step-510000']
OLD={'step-10000','step-30000','step-50000','step-70000','step-80000'}

def save(state):
 p=BASE/'rankme-all-state.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');p.replace(BASE/'rankme-all-state.json')

def score():
 sampling=pd.read_csv(BASE/'sampling.csv').set_index('dataset');catalog=pd.read_csv(ROOT/'CONFIGS/tasks/finetune/tasks.csv').fillna('');rows=[];audits=[]
 previous=pd.read_csv(DETAILS/'raw_metrics.csv').set_index(['dataset','task_id','checkpoint'])
 for name,info in sampling.iterrows():
  d=pd.read_csv(BASE/(name+'.csv'),dtype=str,low_memory=False).fillna('');tasks=catalog[(catalog.dataset==name)&(catalog.enabled.astype(str)=='1')&catalog.modalities.map(lambda x:'va' in x.split('|'))];common=np.ones(len(d),dtype=bool)
  for step in STEPS:
   with np.load(BASE/'features'/f'{name}-{step}.npz',allow_pickle=False) as z:
    assert str(z['sample_sha256'])==info.sample_sha256
    common &= z['valid'] & np.isfinite(z['features']).all(axis=1)
  masks={}
  for t in tasks.itertuples():
   _,mask=labels(d,t);mask &= common;key=np.packbits(mask).tobytes();masks.setdefault(key,{'mask':mask,'tasks':[]})['tasks'].append(t.task_id)
  audits.append({'dataset':name,'sampled':len(d),'common_valid':int(common.sum()),'newly_excluded_vs_previous':int(info.sampled_rows-common.sum())})
  for step in STEPS:
   with np.load(BASE/'features'/f'{name}-{step}.npz',allow_pickle=False) as z:x=z['features']
   for item in masks.values():
    mask=item['mask'];a=x[mask].astype(np.float64);e=np.linalg.eigvalsh(a.T@a) if len(a)>=a.shape[1] else np.linalg.eigvalsh(a@a.T);sing=np.sqrt(np.maximum(e,0));p=sing/sing.sum();p=p[p>0];score=float(np.exp(-np.sum(p*np.log(p))))
    for task in item['tasks']:
     rows.append({'dataset':name,'task_id':task,'checkpoint':step,'n_samples':int(mask.sum()),'rankme':score})
     if step in OLD and int(common.sum())==int(pd.read_csv(BASE/'scores'/(name+'.csv')).n_common_decoded.iloc[0]):
      old=float(previous.loc[(name,task,step),'rankme']);assert np.isclose(score,old,rtol=1e-7,atol=1e-7),(name,task,step,score,old)
   print('scored',name,step,flush=True)
 raw=pd.DataFrame(rows);raw.to_csv(DETAILS/'rankme-all-raw.csv',index=False);taskranks=[];dsranks={};trajectory_ranks=[];trajectory_dsranks={};trajectory_values=[]
 for (ds,task),g in raw.groupby(['dataset','task_id']):
  q=g.set_index('checkpoint').loc[STEPS];r=rankdata(q['rankme']);taskranks.append(r);dsranks.setdefault(ds,[]).append(r);v=q.loc[STEPS[:-1],'rankme'].to_numpy();tr=rankdata(v);trajectory_ranks.append(tr);trajectory_dsranks.setdefault(ds,[]).append(tr);trajectory_values.append((v-v.min())/np.ptp(v))
 avg=np.mean(taskranks,axis=0);davg=np.mean([np.mean(v,axis=0) for v in dsranks.values()],axis=0)
 travg=np.mean(trajectory_ranks,axis=0);trdavg=np.mean([np.mean(v,axis=0) for v in trajectory_dsranks.values()],axis=0)
 table=pd.DataFrame({'checkpoint':STEPS,'experiment':['I3-C1' if s.startswith('I3-') else 'JEPA-e5-latest-C1' for s in STEPS],'mean_task_rank_all_17':avg,'mean_dataset_rank_all_17':davg,'mean_task_rank_trajectory_16':[*travg,np.nan],'mean_dataset_rank_trajectory_16':[*trdavg,np.nan],'tasks':len(taskranks),'datasets':len(dsranks),'common_valid_records':sum(a['common_valid'] for a in audits)}).sort_values('mean_task_rank_all_17',ascending=False)
 table.to_csv(DETAILS/'rankme-all-summary.csv',index=False)
 pd.DataFrame({'checkpoint':STEPS[:-1],'task_mean_rank':travg,'dataset_mean_rank':trdavg}).sort_values('task_mean_rank',ascending=False).to_csv(OUT/'rankme-trajectory16-summary.csv',index=False)
 pd.DataFrame({'checkpoint':STEPS[:-1],'task_equal_minmax_score':np.mean(trajectory_values,axis=0)}).sort_values('task_equal_minmax_score',ascending=False).to_csv(OUT/'rankme-trajectory16-value-score.csv',index=False)
 (DETAILS/'rankme-all-audit.json').write_text(json.dumps({'sampling':'same 20% sample and task masks as sample20; common valid intersection across all listed checkpoints','latest_duplicate':'JEPA-e5 latest.pt model tensors identical to step-80000.pt','datasets':audits},ensure_ascii=False,indent=2)+'\n');print(table.to_string(index=False),flush=True)

def main():
 DETAILS.mkdir(parents=True,exist_ok=True)
 start=time.time();cfg=yaml.safe_load((HERE/'cache/early-stop/audio-video.yaml').read_text());cfg['checkpoints']=[{'id':s,'path':str(I3 if s.startswith('I3-') else TRAJECTORY/(s+'.pt'))} for s in STEPS];config=BASE/'audio-video-all.yaml';config.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
 datasets=pd.read_csv(BASE/'sampling.csv').dataset.tolist();new=[s for s in STEPS if s not in OLD];workers=[];state={'stage':'features','started':start,'workers':[]}
 for i in range(6):
  steps=new[i::6];log=LOG/f'rankme-all-worker-{i}.log';cmd=[sys.executable,str(HERE/'scripts/features20.py'),'--gpu',str(i%2),'--config',str(config),'--checkpoints',*steps,'--datasets',*datasets]
  with log.open('ab',buffering=0) as f:p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=os.environ.copy())
  workers.append(p);state['workers'].append({'pid':p.pid,'gpu':i%2,'steps':steps,'log':str(log)});save(state);print('started',i,steps,flush=True)
 for p in workers:
  rc=p.wait()
  if rc:state['stage']='failed';state['failure']={'pid':p.pid,'exit_code':rc};save(state);raise RuntimeError(state['failure'])
 state['stage']='scoring';state['features_finished']=time.time();save(state);score();state['stage']='complete';state['finished']=time.time();save(state)
if __name__=='__main__':main()
