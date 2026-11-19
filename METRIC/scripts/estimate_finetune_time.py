#!/usr/bin/env python3
"""Rough planning estimates anchored to the observed early-stop VA runs."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.dont_write_bytecode=True
from sample20 import ROOT,read,official_tests

HERE=Path(__file__).resolve().parent.parent
tasks=read(ROOT/'CONFIGS/tasks/finetune/tasks.csv');tasks=tasks[(tasks.enabled=='1')&tasks.modalities.map(lambda x:'va' in x.split('|'))]
registry=read(ROOT/'CONFIGS/datasets.csv').set_index('split_csv')
subject=read('/home/data/sdb/基座模型数据集/3心理数据集/临床/split_new.csv',usecols=['audio_path','Case_id'])
subject=subject[(subject.audio_path!='')&(subject.Case_id!='')].drop_duplicates();n=subject.groupby('audio_path').Case_id.nunique();subject=subject[~subject.audio_path.isin(n[n>1].index)].drop_duplicates('audio_path').set_index('audio_path').Case_id
rows=[]
for ds,g in tasks.groupby('dataset',sort=False):
 p=g.csv_path.iloc[0];d=read(ROOT/p);excluded=official_tests(ds,Path(registry.loc[p,'root_path']),registry.loc[p,'original_csv_path']);d=d[~d.audio_path.isin(excluded)]
 if ds=='3_私有临床数据_LV':d['case']=d.audio_path.map(subject).fillna('')
 for t in g.itertuples():
  z=d[(d[t.label_column]!='')&d[t.label_column+'_split'].isin(['0','1'])];cases=int(z.loc[z.case.ne(''),'case'].nunique()) if ds=='3_私有临床数据_LV' else None
  if ds=='3_私有临床数据_LV':hours=15.1*cases/953;basis='clinical 5-run median scaled by mapped subjects, DSM reference 953; unverified (training instances differ from subjects)'
  elif ds=='3_AVEC2014_LV':hours=np.nan;basis='instance grouping and throughput not benchmarked'
  elif t.task_id=='RAVDESS-emotion':hours=10.3;basis='5-run observed median, 37–83 epochs'
  elif t.task_id=='IEMOCAP-emotion':hours=12.9;basis='5-run observed median on prior disjoint 4995 records'
  else:hours=10.3/40*25*len(z)/2451;basis='25 epochs, RAVDESS clip throughput, proportional to labeled rows; unverified'
  rows.append({'dataset':ds,'task_id':t.task_id,'train_val_labeled_rows_excluding_official_tests':len(z),'clinical_subjects':cases,'estimated_hours':round(hours,1) if np.isfinite(hours) else None,'basis':basis})
out=pd.DataFrame(rows);dest=HERE/'results'/'finetune-time-estimate.csv';out.to_csv(dest,index=False);print(out.to_string(index=False));print('sum of known estimates',out.estimated_hours.sum(), 'tasks',len(out),'unknown',out.estimated_hours.isna().sum());print(dest)
