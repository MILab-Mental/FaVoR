#!/usr/bin/env python3
"""Score datasets as soon as all checkpoint features are complete."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parent;BASE=HERE/'cache/sample20';OUT=BASE/'scores';STATE=BASE/'score-run.json'
STEPS=['step-10000','step-30000','step-50000','step-70000','step-80000']
CLIN='3_私有临床数据_LV'

def birth(pid):
 try:
  v=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if v[0]=='Z' else v[19]
 except (OSError,IndexError):return None

def live(j):return j.get('birth') is not None and birth(j['pid'])==j['birth']
def save(s):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');p.replace(STATE)
def paths(key):
 name,_,step=key.partition('|'); folder=BASE/'score-shards' if step else OUT; stem=name+('-'+step if step else '')
 return [folder/(stem+ext) for ext in ('.csv','-audit.json','-usage.npz')]
def start(key):
 name,_,step=key.partition('|');log=HERE/'logs'/('sample20-score-'+name+('-'+step if step else '')+'.log');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,str(HERE/'scripts/score20.py'),'--dataset',name]+(['--shard','--steps',step] if step else [])
 with log.open('ab',buffering=0) as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 return {'pid':p.pid,'birth':birth(p.pid),'state':'running','log':str(log),'started':time.time()}

def merge_clinical():
 files=[paths(CLIN+'|'+step) for step in STEPS];frames=[pd.read_csv(p[0]) for p in files];meta=[json.loads(p[1].read_text()) for p in files]
 raw=pd.concat(frames,ignore_index=True);assert len(raw)==190 and not raw.duplicated(['task_id','checkpoint']).any()
 assert len({m['sampled_rows'] for m in meta})==len({m['common_valid_rows'] for m in meta})==1
 uses=[np.load(p[2],allow_pickle=False) for p in files];combined={k:np.logical_or.reduce([u[k] for u in uses]) for k in uses[0].files}
 for u in uses:u.close()
 result=dict(meta[0]);result['steps']=STEPS;result['score_wall_seconds']=sum(m['score_wall_seconds'] for m in meta);result['score_wall_seconds_note']='sum of parallel checkpoint shard wall times';result['shared_preprocessing_seconds']=sum(m['shared_preprocessing_seconds'] for m in meta);result['metric_errors']=[e for m in meta for e in m['metric_errors']]
 with (OUT/(CLIN+'-usage.npz')).open('wb') as f:np.savez_compressed(f,**combined)
 (OUT/(CLIN+'-audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 tmp=OUT/(CLIN+'.tmp');raw.to_csv(tmp,index=False);tmp.replace(OUT/(CLIN+'.csv'))

def main():
 lock=(BASE/'score-run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if STATE.exists():s=json.loads(STATE.read_text())
 else:s={'stage':'scoring','started':time.time(),'jobs':{},'errors':[]};save(s)
 old=s['jobs'].get(CLIN)
 if old and not live(old) and not (OUT/(CLIN+'.csv')).exists():s['jobs'].pop(CLIN);s.setdefault('restarts',[]).append({'dataset':CLIN,'reason':'parallel checkpoint shards','time':time.time()});save(s)
 names=pd.read_csv(BASE/'sampling.csv').dataset.tolist()
 keys=[CLIN+'|'+step for step in STEPS]+[name for name in names if name!=CLIN]
 while s['stage']=='scoring':
  for key,j in list(s['jobs'].items()):
   if j['state']!='running' or live(j):continue
   csv,audit,use=paths(key)
   if csv.exists() and audit.exists() and use.exists():
    result=json.loads(audit.read_text());j['state']='done';j['finished']=time.time();j['result']=result;print('finished',key,'errors',len(result['metric_errors']),flush=True)
   else:
    j['state']='failed';s['errors'].append({'dataset':key,'reason':'scoring process ended without complete outputs','log':j['log']});print('failed',key,flush=True)
   save(s)
  active=sum(j['state']=='running' for j in s['jobs'].values())
  for key in keys:
   if active>=6:break
   if key in s['jobs']:continue
   name=key.partition('|')[0]
   if all((BASE/'features'/f'{name}-{step}.npz').is_file() for step in STEPS):
    s['jobs'][key]=start(key);active+=1;save(s);print('started',key,flush=True)
  if all(key in s['jobs'] and s['jobs'][key]['state']=='done' for key in keys):
   merge_clinical()
   s['stage']='scoring_done';s['finished']=time.time();save(s);print('scoring done',flush=True)
   while json.loads((BASE/'run.json').read_text())['stage']=='features':time.sleep(5)
   result=subprocess.run([sys.executable,str(HERE/'scripts/summary20.py')],cwd=ROOT)
   if result.returncode:
    s['stage']='needs_attention';s['errors'].append({'reason':f'summary exited {result.returncode}'})
   else:s['stage']='complete'
   save(s);break
  feature=json.loads((BASE/'run.json').read_text())
  if feature['stage']=='needs_attention' and active==0:
   s['stage']='needs_attention';s['errors'].append({'reason':'feature extraction requires attention'});save(s);break
  if s['errors'] and active==0:
   s['stage']='needs_attention';save(s);break
  time.sleep(15)
if __name__=='__main__':main()
