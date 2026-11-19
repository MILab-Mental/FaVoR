#!/usr/bin/env python3
"""Evaluate centered top singular energy on the fixed 20% VA sample."""
import os,sys,json,time
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata
from score20 import labels
HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parent;BASE=HERE/'cache/sample20';OUT=HERE/'results/sample20';DETAILS=OUT/'details'
STEPS=[f'step-{n}' for n in range(5000,80001,5000)]+['I3-C1-step-510000'];OLD={'step-10000','step-30000','step-50000','step-70000','step-80000'}

def summarize(raw,steps,name):
 vals=[];ranks=[];groups={};all_winners=[]
 for (dataset,task),g in raw.groupby(['dataset','task_id']):
  x=g.set_index('checkpoint').loc[steps,'energy'].to_numpy();assert np.isfinite(x).all()
  if np.ptp(x)==0:continue
  score=(x.max()-x)/np.ptp(x);vals.append(score);ranks.append(rankdata(-x));groups.setdefault(dataset,[]).append(score);all_winners.append(steps[int(np.argmin(x))])
 out=pd.DataFrame({'checkpoint':steps,'task_equal_score':np.mean(vals,axis=0),'dataset_equal_score':np.mean([np.mean(v,axis=0) for v in groups.values()],axis=0),'task_equal_mean_rank':np.mean(ranks,axis=0),'num_task_first':[all_winners.count(s) for s in steps]}).sort_values('task_equal_score',ascending=False)
 out.to_csv((DETAILS if name=='energy-all17-values.csv' else OUT)/name,index=False);print(name,'\n',out.to_string(index=False),flush=True)

def main():
 DETAILS.mkdir(parents=True,exist_ok=True)
 start=time.time();sampling=pd.read_csv(BASE/'sampling.csv').set_index('dataset');tasks=pd.read_csv(ROOT/'CONFIGS/tasks/finetune/tasks.csv').fillna('');previous=pd.read_csv(DETAILS/'raw_metrics.csv').set_index(['dataset','task_id','checkpoint']);rows=[];audit=[]
 for ds,info in sampling.iterrows():
  d=pd.read_csv(BASE/(ds+'.csv'),dtype=str,low_memory=False).fillna('');ts=tasks[(tasks.dataset==ds)&(tasks.enabled.astype(str)=='1')&tasks.modalities.map(lambda x:'va' in x.split('|'))];common=np.ones(len(d),bool)
  for step in STEPS:
   with np.load(BASE/'features'/f'{ds}-{step}.npz',allow_pickle=False) as z:
    assert str(z['sample_sha256'])==info.sample_sha256;common &= z['valid'] & np.isfinite(z['features']).all(axis=1)
  audit.append({'dataset':ds,'sampled':len(d),'common_valid':int(common.sum())});masks={}
  for t in ts.itertuples():
   _,mask=labels(d,t);mask &= common;k=np.packbits(mask).tobytes();masks.setdefault(k,{'mask':mask,'tasks':[]})['tasks'].append(t.task_id)
  prior_valid=int(pd.read_csv(BASE/'scores'/(ds+'.csv')).n_common_decoded.iloc[0])
  for step in STEPS:
   with np.load(BASE/'features'/f'{ds}-{step}.npz') as z:x=z['features']
   for entry in masks.values():
    a=x[entry['mask']].astype(np.float64);a-=a.mean(axis=0);ss=float(np.square(a).sum());gram=a.T@a if len(a)>=a.shape[1] else a@a.T;top=float(eigh(gram,subset_by_index=[len(gram)-1,len(gram)-1],eigvals_only=True)[0]);value=top/ss
    for task in entry['tasks']:
     rows.append({'dataset':ds,'task_id':task,'checkpoint':step,'n_samples':len(a),'energy':value})
     if step in OLD and common.sum()==prior_valid:
      prior=float(previous.loc[(ds,task,step),'top1_singular_energy']);assert np.isclose(value,prior,rtol=1e-8,atol=1e-10),(ds,task,step,value,prior)
   print('scored',ds,step,flush=True)
 raw=pd.DataFrame(rows);raw.to_csv(DETAILS/'energy-all-raw.csv',index=False);summarize(raw,STEPS[:-1],'energy-trajectory16-values.csv');summarize(raw,STEPS,'energy-all17-values.csv');(DETAILS/'energy-all-audit.json').write_text(json.dumps({'formula':'largest squared singular value / sum squared singular values, centered features; lower better','aggregation':'task min-max normalized among candidate checkpoints, lower energy mapped to higher score, then equal task mean','datasets':audit,'seconds':time.time()-start},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
