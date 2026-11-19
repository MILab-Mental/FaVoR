#!/usr/bin/env python3
"""Value-based transfer correlations with observed five-checkpoint FT F1."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE=Path(__file__).resolve().parent.parent;OUT=HERE/'results/sample20';DETAILS=OUT/'details';sys.dont_write_bytecode=True
STEPS=['step-10000','step-30000','step-50000','step-70000','step-80000']
TARGETS=[('DSM-Depression','dsm'),('RAVDESS-emotion','ravdess'),('IEMOCAP-emotion','iemocap')]

def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
 return float(pearsonr(x,y).statistic)

raw=pd.read_csv(DETAILS/'raw_metrics.csv');original=pd.read_csv(OUT/'summary.csv');queue=json.loads((HERE/'cache/early-stop/queue.json').read_text())['jobs'];rows=[];score_rows=[];task_rows=[]
for metric in original.metric:
 task_scores=[];datasets={}
 for (ds,task),g in raw.groupby(['dataset','task_id'],sort=False):
  q=g.set_index('checkpoint').loc[STEPS];values=q[metric].to_numpy(float)
  if not np.isfinite(values).all() or np.ptp(values)==0:continue
  oriented=-values if metric=='top1_singular_energy' else values
  normalized=(oriented-oriented.min())/np.ptp(oriented)
  task_scores.append(normalized);datasets.setdefault(ds,[]).append(normalized)
  for step,value,score in zip(STEPS,values,normalized):task_rows.append({'metric':metric,'dataset':ds,'task_id':task,'checkpoint':step,'metric_value':value,'normalized_value':score})
 if not task_scores:continue
 global_score=np.mean(task_scores,axis=0);dataset_score=np.mean([np.mean(v,axis=0) for v in datasets.values()],axis=0);selected=int(np.argmax(global_score));row={'metric':metric,'tasks':len(task_scores),'datasets':len(datasets),'selected_checkpoint':STEPS[selected],'dataset_equal_selected_checkpoint':STEPS[int(np.argmax(dataset_score))]}
 for i,step in enumerate(STEPS):score_rows.append({'metric':metric,'checkpoint':step,'task_equal_score':global_score[i],'dataset_equal_score':dataset_score[i]})
 for task,key in TARGETS:
  x=raw[raw.task_id==task].set_index('checkpoint').loc[STEPS,metric].to_numpy(float);x=-x if metric=='top1_singular_energy' else x
  y=np.array([queue[f'va/{task}/{step}']['result']['score'] for step in STEPS]);row[key+'_task_pearson']=corr(x,y);row[key+'_global_pearson']=corr(global_score,y);row[key+'_best_f1']=float(y.max());row[key+'_selected_f1']=float(y[selected]);row[key+'_selected_regret']=float(y.max()-y[selected])
 row['mean_task_pearson']=float(np.mean([row[k+'_task_pearson'] for _,k in TARGETS])) if all(np.isfinite(row[k+'_task_pearson']) for _,k in TARGETS) else np.nan
 row['mean_global_pearson']=float(np.mean([row[k+'_global_pearson'] for _,k in TARGETS])) if all(np.isfinite(row[k+'_global_pearson']) for _,k in TARGETS) else np.nan
 rows.append(row)
out=pd.DataFrame(rows).sort_values('mean_global_pearson',ascending=False,na_position='last');out.to_csv(OUT/'pearson-value-summary.csv',index=False);pd.DataFrame(score_rows).to_csv(DETAILS/'pearson-global-scores.csv',index=False);pd.DataFrame(task_rows).to_csv(DETAILS/'pearson-task-values.csv',index=False)
# Raw-value table keeps the original sign, including lower-is-better energy.
raw_table=out[['metric','dsm_task_pearson','ravdess_task_pearson','iemocap_task_pearson']].rename(columns={'dsm_task_pearson':'DSM-Depression','ravdess_task_pearson':'RAVDESS-emotion','iemocap_task_pearson':'IEMOCAP-emotion'}).copy()
columns=[task for task,_ in TARGETS];mask=raw_table.metric.eq('top1_singular_energy');raw_table.loc[mask,columns]*=-1;raw_table['mean']=raw_table[columns].mean(axis=1,skipna=False);raw_table.sort_values('mean',ascending=False,na_position='last').to_csv(OUT/'pearson-raw-task.csv',index=False)
print(out[['metric','dsm_global_pearson','ravdess_global_pearson','iemocap_global_pearson','mean_global_pearson','mean_task_pearson','selected_checkpoint','dsm_selected_f1','ravdess_selected_f1','iemocap_selected_f1']].round(4).to_string(index=False))
