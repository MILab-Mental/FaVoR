#!/usr/bin/env python3
"""Build requested three-task comparison using eight 20% dataset samples."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr,kendalltau
from extra_metrics import EXTRA
import metrics

HERE=Path(__file__).resolve().parent.parent;BASE=HERE/'cache/sample20';OUT=HERE/'results/sample20';DETAILS=OUT/'details';SCORES=BASE/'scores'
STEPS=['step-10000','step-30000','step-50000','step-70000','step-80000']
TARGETS=[('DSM-Depression','dsm'),('RAVDESS-emotion','ravdess'),('IEMOCAP-emotion','iemocap')]
NAMES=list(metrics.METRICS)+list(EXTRA)

def correlations(a,b):
 if not np.isfinite(a).all() or np.ptp(a)==0 or np.ptp(b)==0:return np.nan,np.nan
 return float(spearmanr(a,b).statistic),float(kendalltau(a,b).statistic)

def main():
 DETAILS.mkdir(parents=True,exist_ok=True)
 sampling=pd.read_csv(BASE/'sampling.csv');feature_state=json.loads((BASE/'run.json').read_text());score_state=json.loads((BASE/'score-run.json').read_text());ft=json.loads((HERE/'cache/early-stop/queue.json').read_text())
 if feature_state['stage']!='features_done' or score_state['stage'] not in ('scoring_done','complete'):raise ValueError('features or scoring still incomplete')
 if feature_state['errors'] or score_state['errors'] or ft['stage']!='complete':raise ValueError('run or fine-tuning state not complete')
 datasets=sampling.dataset.tolist();frames=[];audits={};counts={m:0 for m in NAMES}
 for ds in datasets:
  f=pd.read_csv(SCORES/(ds+'.csv')); audit=json.loads((SCORES/(ds+'-audit.json')).read_text());
  expected=int(sampling.set_index('dataset').loc[ds,'tasks'])*len(STEPS)
  if len(f)!=expected or f.duplicated(['task_id','checkpoint']).any():raise ValueError(f'{ds}: incomplete or duplicate result')
  if audit['metric_errors']:raise ValueError(f'{ds}: {len(audit["metric_errors"])} metric errors; inspect audit')
  with np.load(SCORES/(ds+'-usage.npz'),allow_pickle=False) as z:
   for m in NAMES:counts[m]+=int(z[m].sum())
  frames.append(f);audits[ds]=audit
 raw=pd.concat(frames,ignore_index=True);raw.to_csv(DETAILS/'raw_metrics.csv',index=False)
 out=[]
 for m in NAMES:
  all_task=[];grouped={}
  for (ds,task),g in raw.groupby(['dataset','task_id'],sort=False):
   q=g.set_index('checkpoint')
   if not set(STEPS)<=set(q.index):continue
   x=q.loc[STEPS,m].to_numpy(float)
   if not np.isfinite(x).all():continue
   if m=='top1_singular_energy':x=-x
   r=rankdata(x);all_task.append(r);grouped.setdefault(ds,[]).append(r)
  if not all_task:continue
  global_ranks=np.mean(all_task,axis=0);chosen=int(np.flatnonzero(np.isclose(global_ranks,global_ranks.max(),rtol=1e-12,atol=1e-12))[0]);dataset_ranks=np.mean([np.mean(q,axis=0) for q in grouped.values()],axis=0);dataset_chosen=int(np.flatnonzero(np.isclose(dataset_ranks,dataset_ranks.max(),rtol=1e-12,atol=1e-12))[0])
  row={'metric':m,'selected_checkpoint':STEPS[chosen],'dataset_equal_checkpoint':STEPS[dataset_chosen],
       'global_tasks':len(all_task),'global_datasets':len(grouped),'sample_count_per_checkpoint':counts[m],
       'metric_elapsed_seconds_all_5':float(raw[m+'_seconds'].sum()),'metric_checkpoint_tie':int(np.sum(np.isclose(global_ranks,global_ranks.max(),rtol=1e-12,atol=1e-12)))>1}
  pred=[];actual=[];all_ok=True
  for task,key in TARGETS:
   g=raw[raw.task_id==task].set_index('checkpoint')
   if len(g)!=5 or not set(STEPS)<=set(g.index):raise ValueError(f'missing {task}')
   x=g.loc[STEPS,m].to_numpy(float);y=np.array([ft['jobs'][f'va/{task}/{step}']['result']['score'] for step in STEPS]);row[key+'_sample_count']=int(g.loc[STEPS[0],'n_sample']);row[key+'_best_f1']=float(y.max());row[key+'_selected_f1']=float(y[chosen]);row[key+'_selected_regret']=float(y.max()-y[chosen]);row[key+'_best_checkpoint']=STEPS[int(np.argmax(y))]
   if not np.isfinite(x).all() or np.ptp(x)==0:
    row[key+'_spearman']=np.nan;row[key+'_kendall']=np.nan;all_ok=False;continue
   if m=='top1_singular_energy':x=-x
   rho,tau=correlations(x,y);row[key+'_spearman']=rho;row[key+'_kendall']=tau;pred.append(rankdata(x));actual.append(rankdata(y))
  if all_ok:
   a=np.sum(pred,axis=0);b=np.sum(actual,axis=0);row['aggregate_spearman'],row['aggregate_kendall']=correlations(a,b);row['mean_task_spearman']=float(np.mean([row[k+'_spearman'] for _,k in TARGETS]));row['aggregate_rank_regret']=float((b.max()-b[int(np.argmax(a))])/(len(TARGETS)*len(STEPS)))
  else:row.update(aggregate_spearman=np.nan,aggregate_kendall=np.nan,mean_task_spearman=np.nan,aggregate_rank_regret=np.nan)
  out.append(row)
 table=pd.DataFrame(out).sort_values(['aggregate_spearman','aggregate_kendall','aggregate_rank_regret','mean_task_spearman','metric'],ascending=[False,False,True,False,True],na_position='last');table.to_csv(OUT/'summary.csv',index=False)
 meta={'status':'complete','fraction':0.2,'sampling_rule':'20% of eligible train+validation rows per dataset, excluding identified official tests; seed fixed','sampling_counts':sampling[['dataset','eligible_rows','sampled_rows','excluded_identified_official_test_rows']].to_dict('records'),'feature_wall_seconds_excluding_pilot':feature_state['features_done']-feature_state['started'],'pilot_ravdess_seconds':float(np.load(BASE/'features/2_RAVDESS_SV-step-10000.npz')['feature_seconds']),'scoring_wall_seconds':score_state['finished']-score_state['started'],'shared_preprocessing_seconds':sum(x['shared_preprocessing_seconds'] for x in audits.values()),'per_dataset_audits':audits,'finished':time.time()};(DETAILS/'summary-audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(table[['metric','dsm_spearman','ravdess_spearman','iemocap_spearman','aggregate_spearman','aggregate_kendall','selected_checkpoint','sample_count_per_checkpoint','metric_elapsed_seconds_all_5']].to_string(index=False));print('saved',OUT/'summary.csv')
if __name__=='__main__':main()
