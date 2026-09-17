"""Independent rank, table, prediction and metric checks; no generator imports."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import lightgbm as lgb


def check(root,input_path,baseline,support,seed):
    sys.path[:0]=[str(support/'review/H001'),str(support/'units/B002/review')]
    from verify_full_tables import verify_quality
    from check import score_check
    t=np.load(input_path/'run/data/train.npy',allow_pickle=False);d=np.load(input_path/'run/data/dev.npy',allow_pickle=False)
    with np.load(baseline/'run/generator.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
    def req(v,message):
        if not v:raise ValueError(message)
    checked={}
    for method in ('gaussian_rank','beta_rank','beta_wide_rank','independent_rank'):
        folder=root/method;a=np.load(folder/'synthetic.npy',allow_pickle=False)
        with np.load(folder/'generation.npz',allow_pickle=False) as z:latent,ranks,order=z['latent'],z['ranks'],z['order']
        req(np.array_equal(np.sort(order),np.arange(len(t))),'Permutation missing rows')
        arranged=a[np.argsort(order)];rng=np.random.default_rng(seed)
        for g,n in enumerate(m['counts']):
            sl=slice(m['offsets'][g],m['offsets'][g+1]);idx=m['group_indices'][sl];ref=t[idx,:10]
            req(np.all(arranged[sl,10:]==m['contexts'][g]),'Wrong context')
            req(np.array_equal(np.sort(arranged[sl,:10],axis=0),np.sort(ref,axis=0)),'Conditional marginals changed')
            r=ranks[sl]
            for j in range(10):
                req(np.array_equal(np.sort(r[:,j]),np.arange(1,n+1)),'Rank is not permutation')
                by_rank=np.argsort(r[:,j]);req(np.array_equal(ref[by_rank,j],np.sort(ref[:,j])),'Ranks violate source order')
            if method=='gaussian_rank':expected=rng.standard_normal((n,10))@np.linalg.cholesky(m['correlation'][g]).T
            elif method.startswith('beta'):
                f=9. if method=='beta_wide_rank' else 1.;expected=rng.beta(r/f,(n+1-r)/f)
            else:expected=rng.random((n,10))
            req(np.array_equal(expected,latent[sl]),'Sampling replay mismatch')
            for j in range(10):
                v=arranged[sl,j];ix=np.argsort(latent[sl,j],kind='stable');diff=np.diff(latent[sl,j][ix]);outdiff=np.diff(v[ix])
                req((outdiff[diff>0]>=0).all(),'Rank assignment reversed strict scores')
        req(np.array_equal(rng.permutation(len(t)),order),'Output order replay failed')
        q=json.loads((folder/'QUALITY.json').read_text());quality=verify_quality(a,d,t,q)
        results={}
        for family in ('tree','linear'):
            path=folder/family
            with np.load(path/'predictions.npz',allow_pickle=False) as z:y,p=z['labels'],z['probabilities']
            req(np.array_equal(y,d[:,12]),'Wrong evaluation labels');score=json.loads((path/'METRICS.json').read_text())['utility'];score_check(y,p,score)
            if family=='tree':pred=lgb.Booster(model_file=str(path/'model.txt')).predict(d[:,:12],num_threads=2)
            else:
                v=json.loads((path/'model.json').read_text());req(v['converged'],'Incomplete fit')
                x=np.zeros((len(d),54));x[:,:10]=(d[:,:10]-v['mean'])/v['scale'];x[np.arange(len(d)),10+d[:,10]]=1;x[np.arange(len(d)),14+d[:,11]]=1
                z=x@np.asarray(v['coef']).T+v['intercept'];z-=z.max(1,keepdims=True);pred=np.exp(z);pred/=pred.sum(1,keepdims=True)
            req(np.allclose(p,pred,rtol=1e-9,atol=1e-10) and np.array_equal(p.argmax(1),pred.argmax(1)),'Inference check failed')
            results[family]={'balanced_accuracy':score['balanced_accuracy'],'correct':score['correct'],'probability_max_error':float(abs(p-pred).max())}
        checked[method]={'conditional_marginals_exact':True,'sampling_replay_exact':True,'quality':quality,'students':results}
    return {'status':'passed','seed':seed,'checked':checked,'final_test_scored':False,'independence':'Same assistant author; separate checker implementation and process'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Review already exists')
    value=check(a.run,a.input,a.baseline,a.support,a.seed);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
