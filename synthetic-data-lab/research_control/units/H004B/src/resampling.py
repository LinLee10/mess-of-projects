"""Published empirical beta sampler with declared discrete context adapters."""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'H002/src'))
from rank_synthesis import ordinal,assign_columns

def require(ok,msg):
    if not ok:raise ValueError(msg)

def iid_context(x,ids,seed,g):
    x=np.asarray(x);ids=np.asarray(ids)
    require(x.ndim==2 and x.shape[1]==10 and len(x)>0 and x.dtype.kind in 'iu','Nonempty integer numerical context required')
    require(ids.shape==(len(x),) and ids.dtype.kind in 'iu' and (ids>=0).all() and len(np.unique(ids))==len(ids),'Unique aligned integer ids required')
    require(type(seed) is int and seed>=0 and type(g) is int and g>=0,'Nonnegative integer seed and context required')
    n=len(x);r=np.column_stack([ordinal(x[:,j],ids,j) for j in range(10)])
    rng=np.random.default_rng(np.random.SeedSequence([seed,g,3202]));anchors=rng.integers(0,n,size=n)
    latent=rng.beta(r[anchors],n+1-r[anchors]);reference=np.sort(x,axis=0)
    inverse=reference[np.minimum(np.floor(n*latent).astype(np.int64),n-1),np.arange(10)]
    assigned=assign_columns(latent,x,np.arange(n,dtype=np.int64))
    return {'anchor_ids':ids[anchors],'latent':latent,'inverse':inverse,'assigned':assigned}


def dependence_error(x,y):
    x=np.asarray(x);y=np.asarray(y)
    require(x.ndim==2 and y.shape==x.shape and len(x)>0 and x.shape[1]>0,'Exact aligned feature shapes required')
    require(x.dtype.kind in 'iuf' and y.dtype.kind in 'iuf' and np.isfinite(x).all() and np.isfinite(y).all(),'Finite real features required')
    columns=np.flatnonzero(np.std(x,axis=0)>0)
    collapsed=[int(j) for j in columns if np.std(y[:,j])==0]
    if len(columns)<2:return {'mean_absolute_error':0.,'collapsed_synthetic_columns':collapsed,'pairs':0}
    def matrix(a):
        z=rankdata(a[:,columns],axis=0);z=z-z.mean(0);scale=z.std(0)
        z=np.divide(z,scale,out=np.zeros_like(z),where=scale>0)
        return z.T@z/len(z)
    mask=np.triu_indices(len(columns),1)
    return {'mean_absolute_error':float(abs(matrix(x)-matrix(y))[mask].mean()),'collapsed_synthetic_columns':collapsed,'pairs':len(mask[0])}
