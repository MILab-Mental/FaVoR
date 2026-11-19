#!/usr/bin/env python3
"""Reproducible 20% AV sampling, excluding identifiable official test records."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent.parent; ROOT=HERE.parent; OUT=HERE/'cache/sample20'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42; FRACTION=.20


def read(path,**kw):return pd.read_csv(path,dtype=str,low_memory=False,**kw).fillna('')
def official_tests(name,root,source):
    if name=='2_MER2023_SV':
        d=read(source,usecols=['audio_path','split']); mask=d['split'].str.strip().str.lower().str.startswith('test')
    elif name=='3_AVEC2014_LV':
        d=read(source,usecols=['audio_path','split']); mask=d['split'].str.strip().str.lower().str.startswith('test')
    elif name=='2_MER242526_SV':
        cols=['audio_path','25_track1_split','25_track2_split','26_track2_test','26_track3_testperfer','26_track3_stage2_testperfer']
        d=read(source,usecols=cols);mask=d['25_track1_split'].str.strip().str.lower().str.startswith('test')|d['25_track2_split'].str.strip().str.lower().str.startswith('test')
        for c in cols[3:]:mask|=d[c].str.strip().ne('')
    else:return set()
    return set(d.loc[mask,'audio_path'])


def main():
    tasks=read(ROOT/'CONFIGS/tasks/finetune/tasks.csv');tasks=tasks[(tasks.enabled=='1') & tasks.modalities.map(lambda x:'va' in x.split('|'))]
    registry=read(ROOT/'CONFIGS/datasets.csv').set_index('split_csv');subject=read('/home/data/sdb/基座模型数据集/3心理数据集/临床/split_new.csv',usecols=['audio_path','Case_id']);subject=subject[(subject.audio_path!='')&(subject.Case_id!='')].drop_duplicates();cnt=subject.groupby('audio_path').Case_id.nunique();subject=subject[~subject.audio_path.isin(cnt[cnt>1].index)].drop_duplicates('audio_path').set_index('audio_path').Case_id
    summary=[]
    for name,g in tasks.groupby('dataset',sort=False):
        p=g.csv_path.iloc[0];root=Path(registry.loc[p,'root_path']);source=registry.loc[p,'original_csv_path']; cols=['audio_path','video_path']+list(dict.fromkeys(g.label_column.tolist()))+[c+'_split' for c in dict.fromkeys(g.label_column.tolist())]
        d=read(ROOT/p,usecols=lambda c:c in cols);assert set(cols)<=set(d.columns),name
        is_eligible=d[[c for c in d.columns if c.endswith('_split')]].apply(lambda x:x.str.strip().isin(['0','1'])).any(axis=1)
        tests=official_tests(name,root,source);is_test=d.audio_path.isin(tests);eligible=is_eligible & ~is_test & d.audio_path.str.strip().ne('') & d.video_path.str.strip().ne('')
        if name=='3_私有临床数据_LV':
            d['_subject']=d.audio_path.map(subject).fillna('');eligible &= d['_subject'].ne('')
        else:d['_subject']=''
        n=int(eligible.sum());target=int(np.ceil(n*FRACTION));seed=(SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))%(2**32-1);rng=np.random.default_rng(seed);indices=rng.permutation(np.flatnonzero(eligible.to_numpy()));chosen=[];missing=0
        for i in indices:
            r=d.iloc[i]
            if (root/r.audio_path).is_file() and (root/r.video_path).is_file():chosen.append(int(i))
            else:missing+=1
            if len(chosen)==target:break
        if len(chosen)!=target:raise RuntimeError(f'{name}: insufficient existing pairs for requested sample; {len(chosen)}/{target}')
        part=d.iloc[chosen].copy().reset_index(drop=True);groups=part['_subject'].to_numpy() if name=='3_私有临床数据_LV' else np.arange(len(part))
        if len(set(groups))<2:raise ValueError(f'{name}: fewer than two sample groups')
        train,val=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(part,groups=groups));assert not set(groups[train]) & set(groups[val]);part['_probe_split']='train';part.loc[val,'_probe_split']='val';part['_source_row']=np.array(chosen,dtype=int);part['_subject']=part['_subject'].map(lambda x:hashlib.sha256(x.encode()).hexdigest() if x else '')
        part.to_csv(OUT/(name+'.csv'),index=False)
        summary.append({'dataset':name,'csv_rows':len(d),'train_val_union_before_test_exclusion':int(is_eligible.sum()),'excluded_identified_official_test_rows':int((is_eligible & is_test).sum()),'eligible_rows':n,'sampled_rows':len(part),'missing_media_seen':missing,'probe_train_rows':len(train),'probe_val_rows':len(val),'probe_train_groups':len(set(groups[train])),'probe_val_groups':len(set(groups[val])),'grouping':'subject' if name=='3_私有临床数据_LV' else 'sample','media_root':str(root),'tasks':len(g),'seed':seed,'sample_sha256':hashlib.sha256((OUT/(name+'.csv')).read_bytes()).hexdigest()})
        print(name,'eligible',n,'sampled',len(part),'train/val',len(train),len(val),'missing',missing,'test excluded',int((is_eligible&is_test).sum()),flush=True)
    pd.DataFrame(summary).to_csv(OUT/'sampling.csv',index=False)
    (OUT/'sampling.json').write_text(json.dumps({'fraction':FRACTION,'seed':SEED,'source':'union task split 0/1 excluding identifiable official test','clinical_unit':'CSV sentence row; subject-disjoint probe','other_unit':'CSV row; sample-disjoint probe','rows':summary},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
