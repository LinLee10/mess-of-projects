"""One recoverable calibration experiment per seed; no final test evaluation."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from calibration import interpolate, calibrate, benchmark, confusion, rank_map

RAW_SHA='89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def seal(folder):
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='DONE.json'}
    write(folder/'DONE.json',{'files':files,'status':'completed'})

def prepare(raw,out):
    if sha(raw)!=RAW_SHA:raise ValueError('Raw archive digest mismatch')
    with zipfile.ZipFile(raw) as z:
        names=[n for n in z.namelist() if n.endswith('covtype.data.gz')]
        if len(names)!=1:raise ValueError('Unexpected archive schema')
        a=pd.read_csv(io.BytesIO(gzip.decompress(z.read(names[0]))),header=None,dtype=np.int64).to_numpy()
    if a.shape!=(581012,55):raise ValueError('Wrong source dimensions')
    if not (((a[:,10:54]==0)|(a[:,10:54]==1)).all() and (a[:,10:14].sum(1)==1).all() and (a[:,14:54].sum(1)==1).all()):raise ValueError('Illegal original indicators')
    x=np.column_stack([a[:,:10],a[:,10:14].argmax(1),a[:,14:54].argmax(1),a[:,-1]-1]).astype('<i8')
    code=pd.util.hash_pandas_object(pd.DataFrame(x[:,:12]),index=False).to_numpy()%100
    ids={'train':np.flatnonzero(code<60),'dev':np.flatnonzero((code>=60)&(code<80)),'reserved':np.flatnonzero(code>=80)}
    counts={k:len(v) for k,v in ids.items()}
    if counts!={'train':348563,'dev':116314,'reserved':116135}:raise ValueError('Unexpected reconstructed partition')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'membership.npz',**ids)
    train,dev=x[ids['train']],x[ids['dev']]
    np.save(out/'train.npy',train,allow_pickle=False);np.save(out/'dev.npy',dev,allow_pickle=False)
    write(out/'SPLIT.json',{'counts':counts,'raw_sha256':RAW_SHA,'algorithm':'pandas hash of compact int64 predictors only, index=False, modulo 100: <60 training, <80 development, otherwise reserved','pandas':pd.__version__,'historical_identity':'Reconstructed, matching historical counts but no old membership artifact exists','final_test_scored':False,'historical_final_exposure':'unknown','no_arbitrary_cap':True})
    seal(out);return train,dev


def fit_student(a,dev,path,seed):
    path.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    cfg={'n_estimators':150,'num_leaves':63,'learning_rate':.08,'max_depth':-1,'min_child_samples':30,'subsample':1.,'colsample_bytree':1.,'n_jobs':2,'random_state':991,'verbosity':-1,'deterministic':True,'force_col_wise':True}
    model=lgb.LGBMClassifier(**cfg);model.fit(a[:,:12],a[:,12],categorical_feature=[10,11])
    p=model.predict_proba(dev[:,:12]);pred=p.argmax(1)
    np.savez_compressed(path/'predictions.npz',labels=dev[:,12],probabilities=p)
    model.booster_.save_model(str(path/'model.txt'))
    score=confusion(dev[:,12],pred)
    write(path/'METRICS.json',{'utility':score,'config':cfg,'fit_and_predict_seconds':time.monotonic()-start,'training_rows':len(a),'generation_seed':seed,'generation_data_hash':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),'evaluation':'development only, not final'})
    seal(path);return score


def main(raw,out,seed):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'START.json',{'seed':seed,'source_commit':os.environ.get('GITHUB_SHA','local_before_publication'),'run_id':os.environ.get('GITHUB_RUN_ID'),'command':' '.join(os.sys.argv),'status':'started','paid_compute':False})
    for file in Path(__file__).parent.glob('*.py'):
        (out/'source').mkdir(exist_ok=True);shutil.copy2(file,out/'source'/file.name)
    write(out/'ENVIRONMENT.json',{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__})
    train,dev=prepare(raw,out/'data')
    # R002 negative control: constant numerical output. Reorder only records, not the multiset.
    n=len(train);ids=np.arange(n);prototype=np.tile(np.median(train[:,:10],axis=0),(n,1))
    pc=np.column_stack([prototype,train[:,10:13]]).astype(np.int64)
    canonical_order=np.argsort(train[:,0],kind='stable')
    controls={}
    for mode in ['legacy','keyed','midrank']:
        before=calibrate(pc,train,ids,mode=mode,seed=seed)
        after=calibrate(pc[canonical_order],train,ids[canonical_order],mode=mode,seed=seed)
        changed=np.any(before[canonical_order,:10]!=after[:,:10],axis=1)
        controls[mode]={'rows_changed_by_permutation':int(changed.sum()),'total_rows':n,'numerical_unique_rows':len(np.unique(before[:,:10],axis=0)),'elevation_correlation_with_real_row_elevation':float(np.corrcoef(after[:,0],train[canonical_order,0])[0,1]) if mode!='midrank' else None}
        np.savez_compressed(out/f'prototype_{mode}.npz',before=before,permuted_result=after,permutation=canonical_order,ids=ids)
    write(out/'ORDER_CONTROLS.json',controls)
    # All source values are legal; copying deliberately fails authenticity rather than validity.
    ref_metrics=benchmark(train,dev,train);write(out/'REAL_REFERENCE.json',ref_metrics)
    ref=fit_student(train,dev,out/'real_student',seed)
    raw_table,lineage=interpolate(train,seed);np.savez_compressed(out/'generator_lineage.npz',**lineage)
    scores={'real':ref}
    for name in ['raw','pooled_legacy','pooled_keyed','context_keyed']:
        folder=out/name;folder.mkdir();start=time.monotonic()
        if name=='raw':a=raw_table.copy()
        else:a=calibrate(raw_table,train,ids,context=(name=='context_keyed'),mode='legacy' if name=='pooled_legacy' else 'keyed',seed=seed)
        np.save(folder/'synthetic.npy',a,allow_pickle=False)
        m=benchmark(a,dev,train)
        write(folder/'QUALITY.json',m)
        scores[name]=fit_student(a,dev,folder/'student',seed)
        write(folder/'UNIT.json',{'seed':seed,'variant':name,'elapsed_seconds':time.monotonic()-start,'rows':len(a),'input_pool':'same random pair interpolation for every variant','status':'completed'})
        seal(folder);print(json.dumps({'seed':seed,'variant':name,**scores[name],'mean_ks':m['mean_numeric_ks'],'conditional_ks':m['class_balanced_conditional_ks'],'copies':m['copy_count']}),flush=True)
    write(out/'SUMMARY.json',scores);seal(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.raw,a.output,a.seed)
