"""Exploratory conditional distance audit. Reads saved development distances only."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.stats import ks_2samp

def distances_summary(s,r):
    s=np.asarray(s,dtype=float);r=np.asarray(r,dtype=float)
    if s.ndim!=1 or r.ndim!=1 or len(s)==0 or len(r)==0 or not np.isfinite(s).all() or not np.isfinite(r).all() or (s<0).any() or (r<0).any():raise ValueError('Finite nonnegative distances required')
    q=np.array([.01,.05,.25,.5,.75,.95,.99]);cut=np.quantile(r,q)
    return {'synthetic_rows':len(s),'reference_rows':len(r),'quantile_levels':q.tolist(),'reference_quantiles':cut.tolist(),'synthetic_quantiles':np.quantile(s,q).tolist(),'synthetic_below_reference_quantiles':[(s<x).mean().item() for x in cut],'reference_below_own_quantiles':[(r<x).mean().item() for x in cut],'ks_distance':float(ks_2samp(s,r,method='asymp').statistic),'median_ratio':float(np.median(s)/np.median(r)) if np.median(r)>0 else None}

def analyze(run, reference, output):
    d=np.load(reference/'run/data/dev.npy',allow_pickle=False)
    with np.load(run/'real_development_distances.npz',allow_pickle=False) as z:rd=z['distance']
    if len(d)!=len(rd):raise ValueError('Development alignment')
    result={'status':'completed_exploratory','final_test_scored':False,'reference':'fixed real development rows versus fixed training set','thresholds':'real within-context quantiles; not used for generator fitting','methods':{}}
    for name in ['neighbor_raw','neighbor_rank','adaptive_beta']:
        p=run/name;syn=np.load(p/'synthetic.npy',allow_pickle=False)
        with np.load(p/'proximity.npz',allow_pickle=False) as z:sd=z['distance']
        item={'pooled':distances_summary(sd,rd),'classes':{},'contexts':[],'eligibility_sensitivity':{}}
        for c in range(7):item['classes'][str(c+1)]=distances_summary(sd[syn[:,12]==c],rd[d[:,12]==c])
        for ctx in np.unique(syn[:,10:],axis=0):
            sm=np.all(syn[:,10:]==ctx,axis=1);rm=np.all(d[:,10:]==ctx,axis=1)
            row={'context':ctx.tolist(),'training_or_synthetic_rows':int(sm.sum()),'development_rows':int(rm.sum())}
            if rm.any():row['stats']=distances_summary(sd[sm],rd[rm])
            item['contexts'].append(row)
        for minimum in [20,100,500]:
            eligible=[x for x in item['contexts'] if x['development_rows']>=minimum];den=sum(x['training_or_synthetic_rows'] for x in eligible)
            weights=np.array([x['training_or_synthetic_rows']/den for x in eligible])
            stats=[x['stats'] for x in eligible]
            item['eligibility_sensitivity'][str(minimum)]={'contexts':len(eligible),'synthetic_rows':den,'fraction_all_synthetic_rows':den/len(syn),'weighted_ks':float(sum(w*x['ks_distance'] for w,x in zip(weights,stats))),'below_context_q05_fraction':float(sum(w*x['synthetic_below_reference_quantiles'][1] for w,x in zip(weights,stats))),'real_weighted_below_own_q05_fraction':float(sum(w*x['reference_below_own_quantiles'][1] for w,x in zip(weights,stats))),'below_context_median_fraction':float(sum(w*x['synthetic_below_reference_quantiles'][3] for w,x in zip(weights,stats)))}
        result['methods'][name]=item
        print(name,'pooled KS',item['pooled']['ks_distance'],'median ratio',item['pooled']['median_ratio'],'context >=100',item['eligibility_sensitivity']['100'],flush=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':analyze(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
