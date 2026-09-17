"""H001 scoped calibration experiment. All mappings fit real training data only."""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp


def keys_for(ids, column, seed):
    """Independent per feature tie priorities bound to immutable synthetic IDs."""
    x=np.asarray(ids,dtype=np.uint64)^np.uint64(seed)^np.uint64((column+1)*7919)
    with np.errstate(over='ignore'):
        x=(x+np.uint64(0x9e3779b97f4a7c15))
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        x=x^(x>>np.uint64(31))
    return x


def rank_map(values, reference, ids, column=0, seed=0, mode='keyed'):
    values=np.asarray(values);reference=np.asarray(reference)
    if values.ndim!=1 or reference.ndim!=1 or len(reference)==0 or len(values)!=len(ids):raise ValueError('Bad calibration vectors')
    if not np.isfinite(values).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite calibration input')
    if len(np.unique(ids))!=len(ids):raise ValueError('Unique synthetic IDs required')
    if mode not in ('legacy','keyed','midrank'):raise ValueError('Unknown tie rule')
    if not len(values):return values.copy()
    order=np.argsort(values,kind='stable') if mode!='keyed' else np.lexsort((keys_for(ids,column,seed),values))
    positions=(np.arange(len(values))+.5)/len(values)
    if mode=='midrank':
        _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
        centers=(np.cumsum(counts)-counts/2)/len(values)
        return np.interp(centers[inverse]*(len(reference)-1),np.arange(len(reference)),np.sort(reference))
    out=np.empty(len(values),dtype=float)
    out[order]=np.interp(positions*(len(reference)-1),np.arange(len(reference)),np.sort(reference))
    return out


def calibrate(a,train,ids,*,context=False,mode='keyed',seed=0):
    out=a.copy().astype(float)
    if a.shape[1]!=13 or train.shape[1]!=13:raise ValueError('Covertype compact schema required')
    if context:
        groups=np.unique(a[:,10:13],axis=0)
        partitions=[(np.flatnonzero(np.all(a[:,10:13]==g,axis=1)),np.flatnonzero(np.all(train[:,10:13]==g,axis=1))) for g in groups]
    else:partitions=[(np.arange(len(a)),np.arange(len(train)))]
    for si,ti in partitions:
        if len(ti)==0:raise ValueError('No real training support for context')
        for j in range(10):out[si,j]=rank_map(a[si,j],train[ti,j],np.asarray(ids)[si],j,seed,mode)
    out[:,:10]=np.rint(out[:,:10])
    return out.astype(np.int64)


def interpolate(train,seed):
    r=np.random.default_rng(seed);n=len(train)
    # One anchor for every real training record. This preserves natural categorical counts.
    anchor=r.permutation(n);second=np.empty(n,dtype=np.int64)
    groups,inv=np.unique(train[:,10:13],axis=0,return_inverse=True)
    for g in range(len(groups)):
        choices=np.flatnonzero(inv==g);slots=np.flatnonzero(inv[anchor]==g)
        second[slots]=r.choice(choices,len(slots),replace=True)
    weight=r.uniform(.1,.9,(n,1));a=train[anchor].copy()
    a[:,:10]=np.rint((1-weight)*train[anchor,:10]+weight*train[second,:10])
    return a,{'anchor':anchor,'second':second,'weight':weight[:,0]}


def confusion(y,pred,k=7):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or len(y)==0 or y.dtype.kind not in 'iu' or pred.dtype.kind not in 'iu':raise ValueError('Integer label vectors required')
    if (y<0).any() or (y>=k).any() or (pred<0).any() or (pred>=k).any():raise ValueError('Labels out of bounds')
    cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
    if (cm.sum(1)==0).any():raise ValueError('Absent reference class')
    return {'n':len(y),'correct':int(cm.trace()),'accuracy':float(cm.trace()/len(y)),
      'balanced_accuracy':float(np.mean(np.diag(cm)/cm.sum(1))),
      'class_recall':(np.diag(cm)/cm.sum(1)).tolist(),'confusion_matrix':cm.tolist()}


def unique_keys(a):
    a=np.ascontiguousarray(a,dtype='<i8')
    return a.view(np.dtype((np.void,a.shape[1]*8))).reshape(-1)


def hard_valid(a):
    if a.ndim!=2 or a.shape[1]!=13:raise ValueError('Shape mismatch')
    valid=np.isfinite(a).all(1)&(a==np.floor(a)).all(1)
    valid&=(a[:,1]>=0)&(a[:,1]<=360)&(a[:,2]>=0)&(a[:,2]<=90)
    valid&=(a[:,[3,5,9]]>=0).all(1)&(a[:,6:9]>=0).all(1)&(a[:,6:9]<=255).all(1)
    for j,k in [(10,4),(11,40),(12,7)]:valid&=(a[:,j]>=0)&(a[:,j]<k)
    return valid


def benchmark(a,reference,train):
    n=len(a);yk=a[:,12];yr=reference[:,12]
    marginal=[float(ks_2samp(a[:,j],reference[:,j],method='asymp').statistic) for j in range(10)]
    conditional=[];tails=[]
    for c in range(7):
        aa=a[yk==c];rr=reference[yr==c];tt=train[train[:,12]==c]
        if not len(aa) or not len(rr):conditional.append({'class':c,'mean_ks':1.0,'empty':True});continue
        conditional.append({'class':c,'mean_ks':float(np.mean([ks_2samp(aa[:,j],rr[:,j],method='asymp').statistic for j in range(10)]))})
        for j in range(10):
            for q in [.01,.99]:
                t=float(np.quantile(tt[:,j],q));sa=aa[:,j]<t if q<.5 else aa[:,j]>t;ra=rr[:,j]<t if q<.5 else rr[:,j]>t
                tails.append({'class':c,'feature':j,'quantile':q,'threshold':t,'synthetic_count':int(sa.sum()),'real_count':int(ra.sum()),'synthetic_n':len(aa),'real_n':len(rr),'abs_frequency_error':float(abs(sa.mean()-ra.mean()))})
    key=unique_keys(a);tk=np.unique(unique_keys(train));mk=unique_keys(a[:,:12]);tm=np.unique(unique_keys(train[:,:12]))
    quant=np.quantile(train[:,:10],[.1,.5,.9],axis=0)
    def predicates(x):return np.column_stack([x[:,j]>t for j in range(10) for t in quant[:,j]]+[(x[:,12]==c) for c in range(7)]).astype(float)
    pa=predicates(a);pr=predicates(reference);joint=np.abs(pa.T@pa/n-pr.T@pr/len(reference));iu=np.triu_indices(joint.shape[0],1)
    return {'rows':n,'hard_pass_count':int(hard_valid(a).sum()),'hard_pass_fraction':float(hard_valid(a).mean()),
      'numeric_ks_by_feature':marginal,'mean_numeric_ks':float(np.mean(marginal)),
      'conditional_by_class':conditional,'class_balanced_conditional_ks':float(np.mean([x['mean_ks'] for x in conditional])),
      'class_counts':np.bincount(yk,minlength=7).tolist(),'class_prior_tv':float(np.abs(np.bincount(yk,minlength=7)/n-np.bincount(yr,minlength=7)/len(reference)).sum()/2),
      'unique_count':len(np.unique(key)),'copy_count':int(np.isin(key,tk).sum()),'predictor_copy_count':int(np.isin(mk,tm).sum()),
      'query_count':len(iu[0]),'mean_query_error':float(joint[iu].mean()),'max_query_error':float(joint[iu].max()),
      'class_conditional_tail_events':tails,'tail_mean_abs_error':float(np.mean([v['abs_frequency_error'] for v in tails])),
      'scope':'Full table diagnostics. No universal quality score, formal privacy or external validity guarantee.'}
