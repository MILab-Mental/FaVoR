#!/usr/bin/env python3
"""Score one 20% dataset across five checkpoints with shared feature work."""
import argparse,hashlib,json,os,sys,time,warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata,mode,spearmanr
from scipy.special import softmax
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.linear_model import Ridge,RidgeClassifier,LinearRegression,LogisticRegression
from sklearn.metrics import f1_score,mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip
import metrics,extra_metrics as extra

HERE=Path(__file__).resolve().parent.parent;BASE=HERE/'cache/sample20';OUT=BASE/'scores';OUT.mkdir(parents=True,exist_ok=True)
STEPS=['step-10000','step-30000','step-50000','step-70000','step-80000']
NAMES=list(metrics.METRICS)+list(extra.EXTRA)
PROBES={'ridge_probe','linear_probe','knn_probe'}

def labels(d,task):
 col=task.label_column; active=d[col+'_split'].isin(['0','1']) & d[col].notna() & d[col].ne('')
 if task.task_type=='multi_label_classification':
  y=np.zeros((len(d),int(task.num_class)),dtype=np.int8)
  for i,raw in enumerate(d[col].to_numpy()):
   if not active.iloc[i]:continue
   try:
    tokens=[int(t)-1 for t in raw.split('|') if t.strip()]
    if not tokens or min(tokens)<0 or max(tokens)>=y.shape[1]:raise ValueError('outside label range')
    y[i,tokens]=1
   except (ValueError,TypeError):active.iloc[i]=False
  return y,active.to_numpy()
 v=pd.to_numeric(d[col],errors='coerce');active &= np.isfinite(v.to_numpy(float))
 if task.task_type=='classification':
  active &= v.between(1,int(task.num_class)) & (v%1==0)
  return (v.fillna(1).to_numpy(float)-1).astype(int),active.to_numpy()
 return v.fillna(0).to_numpy(float),active.to_numpy()


def evidence(u,s,y,task,d):
 if task=='classification':targets=np.stack([y==c for c in np.unique(y)],axis=1).astype(float)
 elif task=='multi_label_classification':
  targets=y.astype(float);targets=targets[:,targets.min(0)!=targets.max(0)]
 else:targets=np.asarray(y,float).reshape(-1,1)
 if targets.shape[1]==0:return np.nan
 n=len(y);eig=s*s;res=[]
 for t in targets.T:
  p=u.T@t;p2=p*p;residual=max(0.,float(t@t-p2.sum()));alpha=beta=1.
  for _ in range(100):
   ratio=alpha/beta;gamma=float(np.sum(eig/(eig+ratio)));m2=float(np.sum(eig*p2/(eig+ratio)**2));r2=float(np.sum(p2*(ratio/(eig+ratio))**2)+residual)
   na=max(gamma/(m2+1e-12),1e-12);nb=max((n-gamma)/(r2+1e-12),1e-12)
   if abs(na/nb-ratio)/max(ratio,1e-12)<1e-4:alpha,beta=na,nb;break
   alpha,beta=na,nb
  ratio=alpha/beta;m2=float(np.sum(eig*p2/(eig+ratio)**2));r2=float(np.sum(p2*(ratio/(eig+ratio))**2)+residual);ld=np.log(alpha+beta*eig).sum()+(d-len(eig))*np.log(alpha)
  res.append((d*np.log(alpha)+n*np.log(beta)-ld-beta*r2-alpha*m2-n*np.log(2*np.pi))/(2*n))
 return float(np.mean(res))


class Shared:
 def __init__(self,x,idx):
  start=time.perf_counter();self.idx=idx;self.x=np.asarray(x[idx],dtype=np.float64);self.mean=self.x.mean(0);center=self.x-self.mean;self.n,self.d=self.x.shape
  self.u,self.s,_=np.linalg.svd(self.x,full_matrices=False);cu,self.cs,self.vh=np.linalg.svd(center,full_matrices=False)
  prob=self.s/self.s.sum() if self.s.sum()>1e-14 else np.array([]);prob=prob[prob>0];energy=self.cs*self.cs
  self.rankme=float(np.exp(-np.sum(prob*np.log(prob)))) if len(prob) else np.nan
  self.variance=float(np.var(self.x,axis=0).mean());self.energy=float(energy[0]/energy.sum()) if energy.sum()>1e-28 else np.nan
  self.shrink=float(ledoit_wolf_shrinkage(self.x));variance=float(np.square(center).sum()/(self.n*self.d));self.den=np.maximum((1-self.shrink)*energy/self.n+self.shrink*variance,1e-20)
  cu,_=svd_flip(cu,self.vh.copy(),u_based_decision=False);ncomp=min(32,self.n-1,self.d);self.reduced=cu[:,:ncomp]*self.cs[:ncomp]
  self.cka_norm=float(np.linalg.norm(center.T@center)) if self.n>self.d else float(np.linalg.norm(center@center.T));self.center=center
  self.pair_cache=None;self.prep_seconds=time.perf_counter()-start
 def hscore(self,y,task):
  if task=='multi_label_classification':groups=[y[:,j] for j in range(y.shape[1])]
  else:groups=[np.argmax(metrics.strata(y,task,5),axis=1)]
  vals=[]
  for group in groups:
   classes=np.unique(group)
   if len(classes)<2:continue
   value=0.
   for c in classes:
    subset=self.center[group==c];projection=subset.mean(0)@self.vh.T;value+=len(subset)/self.n*np.sum(projection*projection/self.den)
   vals.append((1-self.shrink)*value)
  return float(np.mean(vals)) if vals else np.nan
 def label_cka(self,y,task):
  if task=='classification':_,index=np.unique(y,return_inverse=True);z=np.eye(index.max()+1)[index]
  else:z=np.asarray(y,float).reshape(-1,1)
  z=z-z.mean(0);den=self.cka_norm*np.linalg.norm(z.T@z)
  return float(np.square(self.center.T@z).sum()/den) if den>1e-20 else np.nan
 def pairs(self):
  if self.pair_cache is None:
   # Exact PARC: cache ranked pairwise feature distances once per sample mask.
   z=StandardScaler().fit_transform(self.reduced);z=z-z.mean(axis=1,keepdims=True);norm=np.linalg.norm(z,axis=1)
   if np.any(norm<=1e-20):raise ValueError('constant PCA feature row')
   z=z/norm[:,None];i,j=np.triu_indices(self.n,1);i=i.astype(np.int32);j=j.astype(np.int32);gram=z@z.T;dist=1-gram[i,j];del gram;rank=rankdata(dist);del dist
   self.pair_cache=(i,j,rank,float(rank.mean()),float(np.square(rank-rank.mean()).sum()))
  return self.pair_cache
 def parc(self,y):
  i,j,r,mean,ss=self.pairs();classes,encoded=np.unique(y,return_inverse=True);n=len(r)
  if len(classes)<2 or ss<=0:return np.nan
  if len(classes) in (2,4,8,16):
   # These one-hot correlations have exactly two floating-point values.
   different=(encoded[i]!=encoded[j]);k=int(different.sum())
   if not k or k==n:return np.nan
   return float(100*(r[different].sum()-mean*k)/np.sqrt(ss*k*(1-k/n)))
  # Match the original implementation's floating-point rank ties for other
  # class counts; collapsing almost-equal distances changes PARC's value.
  onehot=np.eye(len(classes))[encoded];target=1-np.corrcoef(onehot);target=target[i,j];target=rankdata(target);target-=target.mean();den=np.sqrt(ss*np.square(target).sum())
  return float(100*np.dot(r-mean,target)/den) if den>0 else np.nan

class Fold:
 def __init__(self,x,train_idx,val_idx):
  start=time.perf_counter();self.train_idx=train_idx;self.val_idx=val_idx;scaler=StandardScaler().fit(x[train_idx]);self.train=scaler.transform(x[train_idx]);self.val=scaler.transform(x[val_idx]);self.neighbors=None;self.prep_seconds=time.perf_counter()-start
 def get_neighbors(self):
  if self.neighbors is None:
   near=NearestNeighbors(n_neighbors=min(5,len(self.train_idx)),metric='cosine',algorithm='brute',n_jobs=1).fit(self.train)
   self.neighbors=near.kneighbors(self.val,return_distance=False)
  return self.neighbors

def crossfit(x,y,active,fold_ids,task,cache,failures,task_id,step):
 n=len(y); names=['ridge_probe','linear_probe']+([] if task=='multi_label_classification' else ['knn_probe']);predictions={k:np.full_like(y,np.nan,dtype=float) for k in names};seconds={k:0. for k in names};extra_prep=0.
 for f in range(5):
  tr=active&(fold_ids!=f);va=active&(fold_ids==f)
  if not va.any():continue
  if tr.sum()<2 or (task=='classification' and len(np.unique(y[tr]))<2):
   failures.append({'task_id':task_id,'checkpoint':step,'metric':'probe','reason':f'fold {f}: insufficient training samples/classes'});continue
  key=hashlib.sha256(np.packbits(tr).tobytes()).hexdigest()+f'-{f}'
  if key not in cache:
   cache[key]=Fold(x,np.flatnonzero(tr),np.flatnonzero((fold_ids==f)));extra_prep+=cache[key].prep_seconds
  fold=cache[key];positions=np.flatnonzero(va[fold.val_idx]);assert len(positions)==va.sum();yt=y[tr]
  for name in names:
   start=time.perf_counter()
   try:
    if name=='knn_probe':
     near=yt[fold.get_neighbors()[positions]]
     pred=near.mean(axis=1) if task=='regression' else mode(near,axis=1,keepdims=False).mode
    else:
     model=(Ridge(alpha=1.) if name=='ridge_probe' else LinearRegression()) if task=='regression' else (RidgeClassifier(alpha=1.) if name=='ridge_probe' else LogisticRegression(C=1.,max_iter=1000,random_state=42))
     if task=='multi_label_classification':model=OneVsRestClassifier(model)
     model.fit(fold.train,yt);pred=model.predict(fold.val[positions])
    predictions[name][va]=pred
   except Exception as e:failures.append({'task_id':task_id,'checkpoint':step,'metric':name,'reason':type(e).__name__+': '+str(e)[:160]})
   seconds[name]+=time.perf_counter()-start
 values={}
 for name,pred in predictions.items():
  if not np.isfinite(pred[active]).all():values[name]=np.nan;continue
  values[name]=-float(mean_absolute_error(y[active],pred[active])) if task=='regression' else float(f1_score(y[active],pred[active],average='macro',zero_division=0))
 return values,seconds,extra_prep

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--steps',nargs='+',default=STEPS);p.add_argument('--pilot',action='store_true');p.add_argument('--shard',action='store_true');args=p.parse_args();name=args.dataset
 if args.shard and len(args.steps)!=1:raise ValueError('a shard must contain exactly one checkpoint')
 info=pd.read_csv(BASE/'sampling.csv').set_index('dataset').loc[name];d=pd.read_csv(BASE/(name+'.csv'),dtype=str,low_memory=False).fillna('');cat=pd.read_csv(HERE.parent/'CONFIGS/tasks/finetune/tasks.csv').fillna('');tasks=cat[(cat.dataset==name)&(cat.enabled.astype(str)=='1')&cat.modalities.map(lambda x:'va' in x.split('|'))];assert len(tasks)==info.tasks
 arrays={};common=np.ones(len(d),bool)
 for step in (args.steps if args.pilot else STEPS):
  pth=BASE/'features'/f'{name}-{step}.npz'
  with np.load(pth,allow_pickle=False) as data:
   assert str(data['sample_sha256'])==info.sample_sha256 and len(data['features'])==len(d)
   features=data['features'];valid=data['valid'] & np.isfinite(features).all(axis=1);common &= valid
   if step in args.steps:arrays[step]=features
 if not common.any():raise ValueError('no common decoded samples')
 total_start=time.perf_counter();rows=[];failures=[];usage={m:np.zeros(len(d),dtype=bool) for m in NAMES};shared_total=0.
 fold_ids=np.full(len(d),-1,dtype=np.int8);ids=np.flatnonzero(common)
 if name=='3_私有临床数据_LV':
  groups=d['_subject'].to_numpy();assert all(groups[ids]);splitter=GroupKFold(n_splits=5);splits=splitter.split(ids,groups=groups[ids])
 else:
  splitter=KFold(n_splits=5,shuffle=True,random_state=int(info.seed));splits=splitter.split(ids)
 for f,(_,v) in enumerate(splits):fold_ids[ids[v]]=f
 assert np.all(fold_ids[ids]>=0)
 if name=='3_私有临床数据_LV':
  for g in np.unique(groups[ids]):assert len(np.unique(fold_ids[(groups==g)&common]))==1
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  for step in args.steps:
   x=arrays[step];cache={};foldcache={}
   for task in tasks.itertuples():
    y,active=labels(d,task);active &= common;idx=np.flatnonzero(active)
    row={'dataset':name,'task_id':task.task_id,'scope':task.scope,'task_type':task.task_type,'checkpoint':step,'n_sample':len(idx),'n_cv_folds':5,'n_sampled_dataset':len(d),'n_common_decoded':int(common.sum())}
    for m in NAMES:row[m]=np.nan;row[m+'_seconds']=0.
    if len(idx)<5 or (task.task_type=='classification' and len(np.unique(y[active]))<2):
     failures.append({'task_id':task.task_id,'checkpoint':step,'metric':'all','reason':'insufficient valid labeled samples/classes'});rows.append(row);continue
    cachekey=hashlib.sha256(np.packbits(active).tobytes()).hexdigest()
    if cachekey not in cache:
     try:cache[cachekey]=Shared(x,idx);shared_total+=cache[cachekey].prep_seconds
     except Exception as e:failures.append({'task_id':task.task_id,'checkpoint':step,'metric':'all','reason':type(e).__name__+': '+str(e)[:160]});rows.append(row);continue
    sh=cache[cachekey];yt=y[active]
    functions={'rankme':lambda:sh.rankme,'feature_variance':lambda:sh.variance,'top1_singular_energy':lambda:sh.energy,'logme':lambda:evidence(sh.u,sh.s,yt,task.task_type,sh.d),'hscore':lambda:sh.hscore(yt,task.task_type)}
    if task.task_type in ('classification','regression'):functions['label_cka']=lambda:sh.label_cka(yt,task.task_type)
    if task.task_type=='classification':functions.update(parc_pca32=lambda:sh.parc(yt),transrate=lambda:extra.transrate(sh.x,yt),gbc_pca32=lambda:extra.gbc(sh.reduced,yt),sfda_pca32=lambda:extra.sfda(sh.reduced,yt))
    for metric,func in functions.items():
     start=time.perf_counter()
     try:
      value=float(func());row[metric]=value
      if np.isfinite(value):usage[metric] |= active
     except Exception as e:failures.append({'task_id':task.task_id,'checkpoint':step,'metric':metric,'reason':type(e).__name__+': '+str(e)[:160]})
     row[metric+'_seconds']=time.perf_counter()-start
    probe_values,probe_times,prep=crossfit(x,y,active,fold_ids,task.task_type,foldcache,failures,task.task_id,step);shared_total+=prep
    for metric,value in probe_values.items():
     row[metric]=value;row[metric+'_seconds']=probe_times[metric]
     if np.isfinite(value):usage[metric] |= active
    rows.append(row);print('scored',name,step,task.task_id,'samples',len(idx),flush=True)
   del cache,foldcache

 raw=pd.DataFrame(rows);suffix='-pilot' if args.pilot else ('-'+args.steps[0] if args.shard else '');output=BASE/'score-shards' if args.shard else OUT;output.mkdir(parents=True,exist_ok=True);dest=output/(name+suffix+'.csv');tmp=dest.with_suffix('.tmp');raw.to_csv(tmp,index=False);tmp.replace(dest)
 with open(output/(name+suffix+'-usage.npz'),'wb') as f:np.savez_compressed(f,**usage)
 meta={'dataset':name,'steps':args.steps,'sampled_rows':len(d),'common_valid_rows':int(common.sum()),'decode_failures_union':int((~common).sum()),'score_wall_seconds':time.perf_counter()-total_start,'shared_preprocessing_seconds':shared_total,'metric_errors':failures};(output/(name+suffix+'-audit.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print('dataset finished',name,'seconds',round(meta['score_wall_seconds'],1),'failures',len(failures),flush=True)
if __name__=='__main__':main()
