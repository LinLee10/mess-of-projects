"""Training neighborhood calibrated rank smoothing; no privacy guarantee."""
import numpy as np
from scipy.spatial import cKDTree
from rank_synthesis import ordinal,assign_columns


def fit_geometry(t,m):
    if t.dtype.kind not in 'iu' or t.shape!=(m['offsets'][-1],13):raise ValueError('Invalid training table')
    scale=np.maximum(t[:,:10].std(0),1e-8)
    neighbors=np.empty((len(t),5),dtype=np.int64);counts=np.ones(len(t),dtype=np.int64)
    loo=np.zeros(len(t));width=np.ones(len(m['counts']));target=np.zeros(len(width));pilot=np.zeros(len(width))
    for g,n in enumerate(m['counts']):
        sl=slice(m['offsets'][g],m['offsets'][g+1]);idx=m['group_indices'][sl];x=t[idx,:10];z=x/scale
        if n==1:neighbors[idx]=idx[:,None];continue
        tree=cKDTree(z);dd,nn=tree.query(z,k=min(6,n),workers=1)
        for i in range(n):
            choices=nn[i][nn[i]!=i][:5]
            if not len(choices):raise ValueError('No distinct neighbor')
            counts[idx[i]]=len(choices);neighbors[idx[i]]=idx[np.resize(choices,5)]
            loo[idx[i]]=np.linalg.norm(z[i]-z[choices[0]])
        target[g]=np.median(loo[idx])
        r=np.column_stack([ordinal(x[:,j],idx,j) for j in range(10)])
        u=np.random.default_rng(24017+g).beta(r,n+1-r)
        probe=assign_columns(u,x,idx)
        pilot[g]=np.median(tree.query(probe/scale,k=1,workers=1)[0])
        if target[g]>0:width[g]=np.clip((target[g]/max(pilot[g],1e-12))**2,1,float(n))
    return {'scale':scale,'neighbors':neighbors,'neighbor_counts':counts,'loo_distance':loo,'bandwidth':width,'pilot_median':pilot,'target_median':target}


def draw(t,m,geometry,seed,method):
    if method not in ('neighbor_raw','neighbor_rank','adaptive_beta'):raise ValueError('Unknown method')
    if type(seed) is not int or seed<0:raise ValueError('Invalid seed')
    rng=np.random.default_rng(seed);n=len(t);table=np.empty_like(t)
    if method.startswith('neighbor'):
        anchor=rng.permutation(n);choice=rng.integers(geometry['neighbor_counts'][anchor]);other=geometry['neighbors'][anchor,choice]
        weight=rng.uniform(.1,.9,n);raw=t[anchor].copy()
        raw[:,:10]=np.rint((1-weight[:,None])*t[anchor,:10]+weight[:,None]*t[other,:10])
        table=raw.copy();latent=raw[:,:10].copy()
        if method=='neighbor_rank':
            for context in m['contexts']:
                rows=np.flatnonzero(np.all(raw[:,10:]==context,axis=1));ids=np.flatnonzero(np.all(t[:,10:]==context,axis=1))
                table[rows,:10]=assign_columns(raw[rows,:10],t[ids,:10],anchor[rows])
        lineage={'anchor':anchor,'other':other,'choice':choice,'weight':weight,'latent':latent}
    else:
        latent=np.empty((n,10));ranks=np.empty((n,10),dtype=np.int64)
        for g,size in enumerate(m['counts']):
            sl=slice(m['offsets'][g],m['offsets'][g+1]);idx=m['group_indices'][sl];x=t[idx,:10]
            r=np.column_stack([ordinal(x[:,j],idx,j) for j in range(10)]);f=geometry['bandwidth'][g]
            u=rng.beta(r/f,(size+1-r)/f)
            table[sl,:10]=assign_columns(u,x,idx);table[sl,10:]=m['contexts'][g]
            latent[sl]=u;ranks[sl]=r
        order=rng.permutation(n);table=table[order];lineage={'latent':latent,'ranks':ranks,'order':order}
    return table,lineage
