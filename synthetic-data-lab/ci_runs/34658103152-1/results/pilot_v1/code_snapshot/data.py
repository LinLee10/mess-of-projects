"""Locally available public benchmarks. No network fetch and no synthetic fixtures."""
from __future__ import annotations
import numpy as np
from sklearn.datasets import load_wine,load_breast_cancer
from sklearn.model_selection import train_test_split
from .common import array_hash,validate_xy

SOURCES={
 'wine':{'url':'https://archive.ics.uci.edu/dataset/109/wine','task':'Wine cultivar classification from 13 chemical measurements','train_cap':60},
 'breast_cancer':{'url':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic','task':'Public diagnostic classification benchmark, not clinical validation','train_cap':120},
}


def make_split(dataset: str,seed: int) -> dict:
    if dataset not in SOURCES: raise ValueError('Unsupported dataset')
    data={'wine':load_wine,'breast_cancer':load_breast_cancer}[dataset]()
    x,y=validate_xy(data.data,data.target)
    ids=np.arange(len(x))
    available,test=train_test_split(ids,test_size=.25,random_state=seed,stratify=y)
    train,dev=train_test_split(available,test_size=.25,random_state=seed+1009,stratify=y[available])
    cap=SOURCES[dataset]['train_cap'];unused=np.array([],dtype=int)
    if len(train)>cap:
        train,unused=train_test_split(train,train_size=cap,random_state=seed+2017,stratify=y[train])
    result={}
    for key,idx in [('train',train),('dev',dev),('test',test),('unused',unused)]:
        arrays={'x':x[idx].copy(),'y':y[idx].copy(),'ids':ids[idx].copy()}
        for a in arrays.values():a.setflags(write=False)
        result[key]=arrays
    result['metadata']={'dataset':dataset,'split_seed':seed,'source':SOURCES[dataset],
        'feature_names':list(data.feature_names),'target_names':[str(v) for v in data.target_names],
        'source_x_sha256':array_hash(x),'source_y_sha256':array_hash(y),
        'counts':{k:len(result[k]['ids']) for k in ('train','dev','test','unused')},
        'split_hashes':{k:array_hash(result[k]['ids']) for k in ('train','dev','test','unused')},
        'split_policy':'Stratified fixed outer split, capped real training budget, no source sharing within a split'}
    return result
