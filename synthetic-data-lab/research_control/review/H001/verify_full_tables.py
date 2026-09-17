"""Independent full table check. No implementation or production scorer imports.

Recomputes every scalar quality field, tail event and query on all rows. Does
not certify causal truth, formal privacy, near copy risk, or final performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def close(x, y, name):
    require(np.isfinite(x) and np.isfinite(y) and abs(x-y) <= 1e-12, f'Mismatch: {name}: {x} != {y}')


def ecdf_distance(x, y):
    x = np.sort(x); y = np.sort(y)
    require(len(x) > 0 and len(y) > 0, 'Empty ECDF')
    support = np.union1d(x, y)
    a = np.searchsorted(x, support, side='right') / len(x)
    b = np.searchsorted(y, support, side='right') / len(y)
    return float(np.abs(a-b).max())


def exact_counts(a, t):
    # Comparison of complete records through structured scalar representations;
    # avoids approximate hashes and does not call the production void-key helper.
    names = [('x'+str(j), np.int64) for j in range(a.shape[1])]
    def records(x):
        out = np.empty(len(x), dtype=names)
        for j in range(x.shape[1]): out['x'+str(j)] = x[:, j]
        return out
    aa, tt = records(a), records(t)
    return int(len(np.unique(aa))), int(np.isin(aa, np.unique(tt)).sum())


def valid_mask(a):
    require(a.ndim == 2 and a.shape[1] == 13, 'Invalid schema')
    ok = np.isfinite(a).all(1) & np.equal(a, np.floor(a)).all(1)
    for c, lo, hi in [(1,0,360),(2,0,90),(6,0,255),(7,0,255),(8,0,255),(10,0,3),(11,0,39),(12,0,6)]:
        ok &= (a[:,c] >= lo) & (a[:,c] <= hi)
    for c in (3,5,9): ok &= a[:,c] >= 0
    return ok


def verify_quality(a, d, t, expected):
    require(a.dtype.kind in 'iu', 'Declared integer output required')
    require(len(a) == expected['rows'], 'Wrong row denominator')
    require(int(valid_mask(a).sum()) == expected['hard_pass_count'], 'Hard validity mismatch')
    close(float(valid_mask(a).mean()), expected['hard_pass_fraction'], 'hard validity fraction')
    distances = [ecdf_distance(a[:,j], d[:,j]) for j in range(10)]
    for j, v in enumerate(distances): close(v, expected['numeric_ks_by_feature'][j], f'KS{j}')
    close(float(np.mean(distances)), expected['mean_numeric_ks'], 'KS mean')
    tails = {(r['class'],r['feature'],r['quantile']):r for r in expected['class_conditional_tail_events']}
    require(len(tails)==140, 'Missing or duplicate tail checks')
    conditional=[]; errors=[]
    for c in range(7):
        aa=a[a[:,12]==c]; dd=d[d[:,12]==c]; tt=t[t[:,12]==c]
        require(len(aa)>0 and len(dd)>0, 'Empty reference class')
        v=float(np.mean([ecdf_distance(aa[:,j],dd[:,j]) for j in range(10)]))
        close(v,expected['conditional_by_class'][c]['mean_ks'],'conditional KS')
        conditional.append(v)
        for j in range(10):
            for q in (.01,.99):
                r=tails[(c,j,q)]; threshold=float(np.quantile(tt[:,j],q))
                close(threshold,r['threshold'],'tail threshold')
                if q<.5:
                    na=int(np.count_nonzero(aa[:,j]<threshold));nd=int(np.count_nonzero(dd[:,j]<threshold))
                else:
                    na=int(np.count_nonzero(aa[:,j]>threshold));nd=int(np.count_nonzero(dd[:,j]>threshold))
                require((na,nd,len(aa),len(dd))==(r['synthetic_count'],r['real_count'],r['synthetic_n'],r['real_n']), 'Tail event count mismatch')
                err=abs(na/len(aa)-nd/len(dd));close(err,r['abs_frequency_error'],'tail event error');errors.append(err)
    close(float(np.mean(conditional)),expected['class_balanced_conditional_ks'],'conditional mean')
    close(float(np.mean(errors)),expected['tail_mean_abs_error'],'tail mean')
    ac=np.bincount(a[:,12],minlength=7);dc=np.bincount(d[:,12],minlength=7)
    require(ac.tolist()==expected['class_counts'],'Class frequency count mismatch')
    close(float(np.abs(ac/len(a)-dc/len(d)).sum()/2),expected['class_prior_tv'],'class TV')
    uc, copies=exact_counts(a,t);_,pc=exact_counts(a[:,:12],t[:,:12])
    require((uc,copies,pc)==(expected['unique_count'],expected['copy_count'],expected['predictor_copy_count']), 'Unique/copy mismatch')
    thresholds=np.quantile(t[:,:10],[.1,.5,.9],axis=0)
    tests=[('num',j,float(thresholds[q,j])) for j in range(10) for q in range(3)] + [('cls',12,c) for c in range(7)]
    def matrix(x):
        return np.column_stack([(x[:,j]>v) if kind=='num' else (x[:,j]==v) for kind,j,v in tests])
    am=matrix(a);dm=matrix(d);query_errors=[]
    for i in range(len(tests)):
        for j in range(i+1,len(tests)):
            u=np.count_nonzero(am[:,i] & am[:,j])/len(a)
            v=np.count_nonzero(dm[:,i] & dm[:,j])/len(d)
            query_errors.append(abs(u-v))
    require(len(query_errors)==expected['query_count'],'Query coverage mismatch')
    close(float(np.mean(query_errors)),expected['mean_query_error'],'query mean')
    close(float(np.max(query_errors)),expected['max_query_error'],'query worst')
    return {'quality_scalars_and_counts_checked':True,'tail_events_checked':len(tails),'joint_queries_checked':len(query_errors), 'rows':len(a)}


def verify(run):
    run=Path(run).resolve();t=np.load(run/'data/train.npy',allow_pickle=False);d=np.load(run/'data/dev.npy',allow_pickle=False)
    groups_t=np.unique(t[:,:12],axis=0);groups_d=np.unique(d[:,:12],axis=0)
    _,overlap=exact_counts(groups_d,groups_t);require(overlap==0,'Predictor identities shared across train and dev')
    with np.load(run/'generator_lineage.npz',allow_pickle=False) as z:
        anchor,second,weight=z['anchor'],z['second'],z['weight']
    require(np.array_equal(np.sort(anchor),np.arange(len(t))),'Anchor coverage incomplete')
    require(np.array_equal(t[anchor,10:13],t[second,10:13]),'Interpolation spans wrong contexts')
    require(((weight>=.1)&(weight<=.9)).all(),'Unexpected interpolation weight')
    recovered=t[anchor].copy();recovered[:,:10]=np.rint((1-weight[:,None])*t[anchor,:10]+weight[:,None]*t[second,:10])
    require(np.array_equal(recovered,np.load(run/'raw/synthetic.npy',allow_pickle=False)), 'Raw generation reconstruction mismatch')
    results={}
    for name in ('raw','pooled_legacy','pooled_keyed','context_keyed'):
        a=np.load(run/name/'synthetic.npy',allow_pickle=False)
        require(np.array_equal(a[:,10:13],recovered[:,10:13]), 'Context association or row order changed')
        results[name]=verify_quality(a,d,t,read(run/name/'QUALITY.json'))
    controls=read(run/'ORDER_CONTROLS.json')
    for mode in ('legacy','keyed','midrank'):
        with np.load(run/f'prototype_{mode}.npz',allow_pickle=False) as z:
            b,p,order=z['before'],z['permuted_result'],z['permutation']
        changed=int(np.count_nonzero(np.any(b[order,:10]!=p[:,:10],axis=1)))
        require(changed==controls[mode]['rows_changed_by_permutation'], 'Permutation mismatch')
    return {'status':'passed','production_modules_imported':False,'source_training_rows':len(t),'development_rows':len(d),'train_dev_predictor_overlap':overlap,'interpolation_lineage_reconstruction':'exact','variants':results,'scope':'Full hard-rule, marginal, class conditional, tail, joint query, diversity, copy, input identity and order control recomputation; no final test evaluated'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists() and not a.output.resolve().is_relative_to(a.run.resolve()), 'Use new output outside evidence')
    result=verify(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
