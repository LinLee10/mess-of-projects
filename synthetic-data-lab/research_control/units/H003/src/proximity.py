"""Full row contextual distance diagnostics, not a privacy guarantee."""
import numpy as np
from scipy.spatial import cKDTree


def nearest(train,query,scale):
    train=np.asarray(train);query=np.asarray(query);scale=np.asarray(scale)
    if train.ndim!=2 or query.ndim!=2 or train.shape[1]!=13 or query.shape[1]!=13 or len(train)==0 or scale.shape!=(10,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid arrays or scales')
    if train.dtype.kind not in 'iu' or query.dtype.kind not in 'iu':raise ValueError('Integer compact tables required')
    distances=np.full(len(query),np.inf);identity=np.full(len(query),-1,dtype=np.int64)
    contexts,ti=np.unique(train[:,10:],axis=0,return_inverse=True)
    for g,context in enumerate(contexts):
        qi=np.flatnonzero(np.all(query[:,10:]==context,axis=1))
        if not len(qi):continue
        ii=np.flatnonzero(ti==g);tree=cKDTree(train[ii,:10]/scale)
        dd,nn=tree.query(query[qi,:10]/scale,k=1,workers=1)
        distances[qi]=dd;identity[qi]=ii[nn]
    return distances,identity


def summarize(distances,reference):
    d=np.asarray(distances);r=np.asarray(reference);valid=np.isfinite(d);rv=np.isfinite(r)
    if d.ndim!=1 or r.ndim!=1 or len(d)==0 or not rv.any() or (d<0).any() or (r<0).any() or np.isnan(d).any() or np.isnan(r).any():raise ValueError('Invalid distances')
    threshold=float(np.quantile(r[rv],.05))
    return {'rows':len(d),'unmatched_context_rows':int((~valid).sum()),'zero_distance_rows':int((d==0).sum()),'real_development_q05_distance':threshold,
            'strictly_below_real_q05_count':int((d<threshold).sum()),'strictly_below_real_q05_fraction':float((d<threshold).mean()),
            'finite_distance_quantiles':np.quantile(d[valid],[0,.01,.05,.5,.95,1]).tolist() if valid.any() else None,
            'reference_finite_rows':int(rv.sum()),'reference_unmatched_rows':int((~rv).sum()),'reference_median_distance':float(np.median(r[rv])),
            'meaning':'Euclidean distance to the closest real training row inside identical wilderness, soil and cover context; each feature scaled by real training standard deviation. No formal privacy claim.'}
