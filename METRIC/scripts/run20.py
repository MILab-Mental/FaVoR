#!/usr/bin/env python3
"""Run 20% eight-dataset extraction on two GPUs; persist state and logs."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parent;BASE=HERE/'cache/sample20';STATE=BASE/'run.json'
A=['3_私有临床数据_LV','2_CREMA-D_SV','3_AVEC2014_LV']
B=['2_EmotionTalk_SV','2_IEMOCAP_SV','2_MER2023_SV','2_MER242526_SV','2_RAVDESS_SV']
PLAN=[('g0-s10000',0,'step-10000',A+B),('g0-s50000',0,'step-50000',A+B),('g0-s80000a',0,'step-80000',A),('g1-s30000',1,'step-30000',A+B),('g1-s70000',1,'step-70000',A+B),('g1-s80000b',1,'step-80000',B)]

def birth(pid):
 try:
  v=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if v[0]=='Z' else v[19]
 except (OSError,IndexError):return None

def live(job):return job.get('birth') is not None and birth(job['pid'])==job['birth']
def save(s):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');p.replace(STATE)
def start(name,gpu,step,datasets):
 log=HERE/'logs'/('sample20-'+name+'.log');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,str(HERE/'scripts/features20.py'),'--gpu',str(gpu),'--datasets',*datasets,'--checkpoints',step,'--batch-size','8','--workers','4']
 with log.open('ab',buffering=0) as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 return {'pid':p.pid,'birth':birth(p.pid),'state':'running','gpu':gpu,'checkpoint':step,'datasets':datasets,'log':str(log),'started':time.time()}

def main():
 lock=(BASE/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if STATE.exists():s=json.loads(STATE.read_text())
 else:s={'stage':'features','started':time.time(),'jobs':{},'errors':[]};save(s)
 while s['stage']=='features':
  for name,gpu,step,datasets in PLAN:
   if name not in s['jobs']:
    s['jobs'][name]=start(name,gpu,step,datasets);save(s);print('started',name,flush=True);time.sleep(2)
  for name,gpu,step,datasets in PLAN:
   j=s['jobs'][name]
   if j['state']!='running' or live(j):continue
   missing=[d for d in datasets if not (BASE/'features'/f'{d}-{step}.npz').is_file()]
   if missing:j['state']='failed';j['missing']=missing;s['errors'].append({'job':name,'missing':missing});print('failed',name,missing,flush=True)
   else:j['state']='done';j['finished']=time.time();print('finished',name,flush=True)
   save(s)
  if all(j['state']!='running' for j in s['jobs'].values()):
   if s['errors']:s['stage']='needs_attention';save(s);break
   s['stage']='features_done';s['features_done']=time.time();save(s);print('features done',flush=True);break
  time.sleep(15)
if __name__=='__main__':main()
