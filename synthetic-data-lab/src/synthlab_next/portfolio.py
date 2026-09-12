"""Student informed, support screened generator portfolio and factorial ablations.

This is an original local synthesis of known selection ideas, not an assertion
of new scientific priority. It selects from existing fitted generator pools;
it does not discover new information or retrain a frontier language model.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .common import validate_xy,validate_labels,array_hash


def utility_weights(losses,counts,temperature=.3,prior_strength=20.,floor=.1):
    losses=np.asarray(losses,dtype=float);counts=np.asarray(counts,dtype=float)
    if losses.ndim!=2 or min(losses.shape)<1 or counts.shape!=(losses.shape[1],) or not np.isfinite(losses).all() or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Invalid development losses or counts')
    if not np.isfinite([temperature,prior_strength,floor]).all() or temperature<=0 or prior_strength<=0 or not 0<floor<1:raise ValueError('Invalid portfolio regularization')
    score=-losses/temperature;score-=score.max(0,keepdims=True)
    p=np.exp(score);p/=p.sum(0,keepdims=True)
    shrink=counts/(counts+prior_strength);m=losses.shape[0]
    p=shrink*p+(1-shrink)/m
    return (1-floor)*p+floor/m


def support_mask(real,generated,lower_multiplier=.02,upper_multiplier=1.5):
    real=np.asarray(real,dtype=float);generated=np.asarray(generated,dtype=float)
    if real.ndim!=2 or generated.ndim!=2 or generated.shape[1]!=real.shape[1] or len(real)<2 or not np.isfinite(real).all() or not np.isfinite(generated).all():raise ValueError('Invalid support comparison')
    if not np.isfinite([lower_multiplier,upper_multiplier]).all() or lower_multiplier<0 or upper_multiplier<=0:raise ValueError('Invalid support thresholds')
    scale=np.maximum(real.std(0),1e-8);z=real/scale
    knn=NearestNeighbors(n_neighbors=2).fit(z)
    within=knn.kneighbors(z,return_distance=True)[0][:,1]
    # Calibrated exclusively on training rows. This is a support heuristic, not a privacy guarantee.
    lower=max(1e-10,lower_multiplier*float(np.quantile(within,.05)))
    upper=max(lower*2,upper_multiplier*float(np.quantile(within,.95)))
    distances=knn.kneighbors(generated/scale,n_neighbors=1,return_distance=True)[0][:,0]
    keep=(distances>lower)&(distances<=upper)
    return keep,{'lower':lower,'upper':upper,'accepted':int(keep.sum()),'total':len(keep),
        'closest_distance':float(distances.min()) if len(distances) else None,
        'privacy_guarantee':False,'calibration':'Class conditional training leave one out nearest neighbor distances'}


def select_portfolio(pools,train_x,train_y,losses,counts,labels,*,seed,weighted,gated):
    train_x,train_y=validate_xy(train_x,train_y);classes=len(np.unique(train_y));labels=validate_labels(labels,classes)
    names=list(pools)
    if not names:raise ValueError('Empty generator portfolio')
    weights=utility_weights(losses,counts) if weighted else np.full((len(names),classes),1/len(names))
    if weights.shape!=(len(names),classes):raise ValueError('Loss matrix does not match portfolio order')
    matrices=[];targets=[];sources=[];offsets=[];gates={};global_offset=0
    for i,name in enumerate(names):
        x,y=validate_xy(*pools[name])
        if x.shape[1]!=train_x.shape[1] or len(np.unique(y))!=classes:raise ValueError('Mismatched source pool')
        keep=np.ones(len(y),dtype=bool);gates[name]={}
        if gated:
            for c in range(classes):
                idx=np.flatnonzero(y==c);mask,diagnostic=support_mask(train_x[train_y==c],x[idx])
                keep[idx]=mask;gates[name][str(c)]=diagnostic
        matrices.append(x[keep]);targets.append(y[keep]);sources.append(np.full(keep.sum(),i))
        offsets.append(np.arange(global_offset,global_offset+len(y))[keep]);global_offset+=len(y)
    x=np.concatenate(matrices);y=np.concatenate(targets);source=np.concatenate(sources);ids=np.concatenate(offsets)
    # Deduplicate exact feature and label tuples before curation, regardless of source.
    seen=set();unique=[]
    for i in range(len(y)):
        key=(int(y[i]),x[i].tobytes())
        if key not in seen:seen.add(key);unique.append(i)
    unique=np.asarray(unique,dtype=int);x=x[unique];y=y[unique];source=source[unique];ids=ids[unique]
    rng=np.random.default_rng(seed);out=np.empty((len(labels),train_x.shape[1]));selected=np.empty(len(labels),dtype=int)
    for c in range(classes):
        slots=np.flatnonzero(labels==c);candidates=np.flatnonzero(y==c)
        if len(candidates)<len(slots):raise RuntimeError('Insufficient unique supported samples; no unreported fallback')
        per_source=np.bincount(source[candidates],minlength=len(names));p=np.array([weights[s,c]/per_source[s] for s in source[candidates]])
        p/=p.sum();chosen=rng.choice(candidates,len(slots),replace=False,p=p)
        out[slots]=x[chosen];selected[slots]=chosen
    meta={'weighted':weighted,'gated':gated,'method_order':names,'class_method_weights':weights.tolist(),
        'support_gates':gates,'selected_sources':[names[source[j]] for j in selected],
        'selected_pool_indices':ids[selected].tolist(),'unique_candidate_count':len(y),
        'x_sha256':array_hash(out),'y_sha256':array_hash(labels),
        'scope':'Development informed portfolio selection, not a new pretrained generator',
        'cost_includes':'All component generator fits and candidate draws, plus development probe student training'}
    return out,labels.copy(),meta
