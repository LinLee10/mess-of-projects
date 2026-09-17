"""Independent sampling, table, neighbor, coefficient and metric checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import KDTree
import lightgbm as lgb


def require(ok,msg):
    if not ok:raise ValueError(msg)


def keyed_rank_assign(u,x,ids):
    a=np.empty_like(x)
    for j in range(x.shape[1]):
        h=np.asarray(ids,dtype=np.uint64)^np.uint64((j+1)*7919)^np.uint64(217)
        with np.errstate(over='ignore'):
            h=h+np.uint64(0x9e3779b97f4a7c15);h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);h=h^(h>>np.uint64(31))
        a[np.lexsort((h,u[:,j])),j]=np.sort(x[:,j])
    return a


def ordinal(x,ids):
    a=keyed_rank_assign(x,np.tile(np.arange(1,len(x)+1)[:,None],(1,10)),ids)
    return a


def check(a):
    sys.path[:0]=[str(a.support/'review/H001'),str(a.support/'units/B002/review')]
    from verify_full_tables import verify_quality
    from check import score_check
    t=np.load(a.input/'run/data/train.npy',allow_pickle=False);d=np.load(a.input/'run/data/dev.npy',allow_pickle=False)
    with np.load(a.baseline/'run/generator.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
    with np.load(a.run/'geometry.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    scale=np.maximum(t[:,:10].std(0),1e-8);require(np.array_equal(scale,g['scale']),'Scale mismatch')
    trees=[]
    for c,n in enumerate(m['counts']):
        sl=slice(m['offsets'][c],m['offsets'][c+1]);idx=m['group_indices'][sl];x=t[idx,:10];tree=KDTree(x/scale);trees.append((idx,tree))
        if n==1:
            require(g['bandwidth'][c]==1 and g['loo_distance'][idx[0]]==0,'Singleton policy');continue
        dd,_=tree.query(x/scale,k=min(n,6))
        require(np.allclose(dd[:,1],g['loo_distance'][idx],rtol=0,atol=1e-10),'Leave one out distances mismatch')
        require(np.all(t[g['neighbors'][idx],10:]==t[idx,None,10:]),'Neighbor crossed context')
        require(not np.any(g['neighbors'][idx]==idx[:,None]),'Self neighbor')
        distances=np.linalg.norm((t[g['neighbors'][idx],:10]-x[:,None,:])/scale,axis=2)
        require((distances<=dd[:,-1,None]+1e-10).all(),'Not among nearest neighbors')
        r=ordinal(x,idx);u=np.random.default_rng(24017+c).beta(r,n+1-r)
        probe=keyed_rank_assign(u,x,idx);pm=float(np.median(tree.query(probe/scale,k=1)[0]));target=float(np.median(dd[:,1]))
        require(abs(pm-g['pilot_median'][c])<1e-10 and abs(target-g['target_median'][c])<1e-10,'Pilot or target mismatch')
        width=np.clip((target/max(pm,1e-12))**2,1,float(n)) if target>0 else 1.
        require(abs(width-g['bandwidth'][c])<1e-8,'Bandwidth fitting mismatch')
    with np.load(a.run/'real_development_distances.npz',allow_pickle=False) as z:real_distance=z['distance']
    recovered=np.full(len(d),np.inf)
    for c,(idx,tree) in enumerate(trees):
        slots=np.flatnonzero(np.all(d[:,10:]==m['contexts'][c],axis=1))
        if len(slots):recovered[slots]=tree.query(d[slots,:10]/scale,k=1)[0][:,0]
    require(np.allclose(recovered,real_distance,rtol=0,atol=1e-10),'Real reference distances mismatch')
    checked={}
    for method in ('neighbor_raw','neighbor_rank','adaptive_beta'):
        p=a.run/method;table=np.load(p/'synthetic.npy',allow_pickle=False)
        with np.load(p/'generation.npz',allow_pickle=False) as z:l={k:z[k] for k in z.files}
        rng=np.random.default_rng(a.seed)
        if method.startswith('neighbor'):
            anchor=rng.permutation(len(t));choice=rng.integers(g['neighbor_counts'][anchor]);other=g['neighbors'][anchor,choice];w=rng.uniform(.1,.9,len(t))
            require(np.array_equal(anchor,l['anchor']) and np.array_equal(other,l['other']) and np.array_equal(w,l['weight']),'Neighbor sampling replay')
            expected=t[anchor].copy();expected[:,:10]=np.rint((1-w[:,None])*t[anchor,:10]+w[:,None]*t[other,:10])
            if method=='neighbor_rank':
                for c,idx_tree in enumerate(trees):
                    ids,_=idx_tree;slots=np.flatnonzero(np.all(expected[:,10:]==m['contexts'][c],axis=1))
                    expected[slots,:10]=keyed_rank_assign(expected[slots,:10],t[ids,:10],anchor[slots])
        else:
            expected=np.empty_like(t)
            for c,n in enumerate(m['counts']):
                sl=slice(m['offsets'][c],m['offsets'][c+1]);idx=m['group_indices'][sl];r=ordinal(t[idx,:10],idx);f=g['bandwidth'][c]
                require(np.array_equal(r,l['ranks'][sl]),'Source ranks changed')
                u=rng.beta(r/f,(n+1-r)/f);require(np.array_equal(u,l['latent'][sl]),'Beta replay changed')
                expected[sl,:10]=keyed_rank_assign(u,t[idx,:10],idx);expected[sl,10:]=m['contexts'][c]
            order=rng.permutation(len(t));require(np.array_equal(order,l['order']),'Order mismatch');expected=expected[order]
        require(np.array_equal(expected,table),'Independent table reconstruction failed')
        quality=verify_quality(table,d,t,json.loads((p/'QUALITY.json').read_text()))
        with np.load(p/'proximity.npz',allow_pickle=False) as z:saved_dist=z['distance']
        recomputed=np.full(len(table),np.inf)
        for c,(idx,tree) in enumerate(trees):
            slots=np.flatnonzero(np.all(table[:,10:]==m['contexts'][c],axis=1))
            if len(slots):recomputed[slots]=tree.query(table[slots,:10]/scale,k=1)[0][:,0]
            if method!='neighbor_raw':require(np.array_equal(np.sort(table[slots,:10],axis=0),np.sort(t[idx,:10],axis=0)),'Changed conditional marginals')
        require(np.allclose(saved_dist,recomputed,rtol=0,atol=1e-10),'Full nearest distances mismatch')
        pr=json.loads((p/'PROXIMITY.json').read_text());near=int((recomputed<np.quantile(recovered[np.isfinite(recovered)],.05)).sum())
        require(pr['strictly_below_real_q05_count']==near and pr['rows']==len(table),'Proximity count mismatch')
        scores={}
        for family in ('tree','linear'):
            folder=p/family
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:y,prob=z['labels'],z['probabilities']
            require(np.array_equal(y,d[:,12]),'Wrong development labels');score=json.loads((folder/'METRICS.json').read_text())['utility'];score_check(y,prob,score)
            if family=='tree':prediction=lgb.Booster(model_file=str(folder/'model.txt')).predict(d[:,:12],num_threads=2)
            else:
                mm=json.loads((folder/'model.json').read_text());require(mm['converged'],'Incomplete linear fit')
                x=np.zeros((len(d),54));x[:,:10]=(d[:,:10]-mm['mean'])/mm['scale'];x[np.arange(len(d)),10+d[:,10]]=1;x[np.arange(len(d)),14+d[:,11]]=1
                z=x@np.asarray(mm['coef']).T+mm['intercept'];z-=z.max(1,keepdims=True);prediction=np.exp(z);prediction/=prediction.sum(1,keepdims=True)
            require(np.allclose(prob,prediction,rtol=1e-9,atol=1e-10) and np.array_equal(prob.argmax(1),prediction.argmax(1)),'Inference mismatch');scores[family]=score
        checked[method]={'students':scores,'quality':quality,'near_reference_count':near,'generation_reconstruction_exact':True}
    return {'status':'passed','checked':checked,'independent_nearest_algorithm':'sklearn KDTree versus production scipy cKDTree','final_test_scored':False,'independence':'Same assistant author, separate implementation and process'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Existing audit')
    r=check(a);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
