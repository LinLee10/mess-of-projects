"""Conditional marginal locked dependence experiment.

Empirical beta smoothing and rank rearrangement are established techniques.
This local combination is an experiment, not a scientific priority claim.
"""
import numpy as np


def keys(ids,j):
    x=np.asarray(ids,dtype=np.uint64)^np.uint64((j+1)*7919)^np.uint64(217)
    with np.errstate(over='ignore'):
        x=x+np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return x^(x>>np.uint64(31))


def ordinal(values,ids,j):
    if len(values)!=len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Unique aligned identities required')
    order=np.lexsort((keys(ids,j),values))
    rank=np.empty(len(order),dtype=np.int64);rank[order]=np.arange(1,len(order)+1)
    return rank


def assign_columns(latent,reference,ids=None):
    if latent.shape!=reference.shape or not np.isfinite(latent).all():raise ValueError('Invalid latent array')
    ids=np.arange(len(latent)) if ids is None else np.asarray(ids)
    if len(ids)!=len(latent) or len(np.unique(ids))!=len(ids):raise ValueError('Unique rank assignment identities required')
    out=np.empty(reference.shape,dtype=np.int64)
    for j in range(out.shape[1]):out[np.lexsort((keys(ids,j),latent[:,j])),j]=np.sort(reference[:,j])
    return out


def generate(train,model,seed,method):
    if method not in ('gaussian_rank','beta_rank','beta_wide_rank','independent_rank'):raise ValueError('Unknown method')
    if type(seed) is not int or seed<0:raise ValueError('Invalid seed')
    if train.shape!=(model['offsets'][-1],13) or train.dtype.kind not in 'iu':raise ValueError('Invalid source schema')
    rng=np.random.default_rng(seed);result=np.empty_like(train);latent=np.empty((len(train),10));ranks=np.empty((len(train),10),dtype=np.int64)
    for g,n in enumerate(model['counts']):
        sl=slice(model['offsets'][g],model['offsets'][g+1]);idx=model['group_indices'][sl]
        a=train[idx,:10]
        r=np.column_stack([ordinal(a[:,j],idx,j) for j in range(10)])
        if method=='gaussian_rank':
            u=rng.standard_normal((n,10))@np.linalg.cholesky(model['correlation'][g]).T
        elif method.startswith('beta'):
            spread=9. if method=='beta_wide_rank' else 1.
            u=rng.beta(r/spread,(n+1-r)/spread)
        else:u=rng.random((n,10))
        result[sl,:10]=assign_columns(u,a,idx);result[sl,10:]=model['contexts'][g]
        latent[sl]=u;ranks[sl]=r
    order=rng.permutation(len(train))
    return result[order], {'latent':latent,'ranks':ranks,'order':order}
