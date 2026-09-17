"""B002 full data reference; immutable input, staged outputs, no final holdout read."""
import argparse, hashlib, json, os, platform, shutil, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from copula import ContextCopula, data_digest


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def verify_input(root):
    manifest=json.loads((root/'ARTIFACT_SHA256.json').read_text())
    if not manifest:raise ValueError('Empty input manifest')
    for rel,h in manifest.items():
        p=root/rel
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Input corruption: '+rel)
    return sha(root/'ARTIFACT_SHA256.json')


def fit_linear(a,dev,out):
    from second_learner import encode, summarize
    out.mkdir();start=time.monotonic()
    mean=a[:,:10].mean(0);scale=np.maximum(a[:,:10].std(0),1e-8)
    x=encode(a,mean,scale);dx=encode(dev,mean,scale)
    cfg={'C':1.,'solver':'lbfgs','max_iter':1000,'tol':1e-5}
    model=LogisticRegression(**cfg)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');model.fit(x,a[:,12])
    convergence=not any(issubclass(w.category,ConvergenceWarning) for w in ws)
    p=model.predict_proba(dx)
    np.savez_compressed(out/'predictions.npz',labels=dev[:,12],probabilities=p)
    write(out/'model.json',{'mean':mean.tolist(),'scale':scale.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),'configuration':cfg,'iterations':model.n_iter_.tolist(),'converged':convergence,'warnings':[str(w.message) for w in ws]})
    scores=summarize(dev[:,12],p)
    write(out/'METRICS.json',{'utility':scores,'fit_and_predict_seconds':time.monotonic()-start,'training_rows':len(a),'input_sha256':data_digest(a),'evaluation':'development only'})
    if not convergence:raise RuntimeError('Linear optimization failed to converge')
    return scores


def main(args):
    root=args.input.resolve();out=args.output.resolve()
    if out.exists() or out.is_relative_to(root):raise ValueError('Use new output outside immutable input')
    out.mkdir(parents=True)
    write(out/'START.json',{'seed':args.seed,'source_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),'command':sys.argv,'status':'started','unit':'B002','final_test_scored':False})
    try:
        digest=verify_input(root)
        sys.path[:0]=[str(args.support/'units/H001/src'),str(args.support/'units/H001C/src')]
        from run_h001 import fit_student
        from calibration import benchmark
        t=np.load(root/'run/data/train.npy',allow_pickle=False);d=np.load(root/'run/data/dev.npy',allow_pickle=False)
        if t.shape!=(348563,13) or d.shape!=(116314,13):raise ValueError('Incorrect full data dimensions')
        write(out/'INPUT.json',{'input_manifest_sha256':digest,'training_sha256':data_digest(t),'development_sha256':data_digest(d),'training_rows':len(t),'development_rows':len(d),'reserved_rows_scored':0})
        write(out/'ENVIRONMENT.json',{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'thread_controls':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},'cpu_only':True})
        start=time.monotonic();model=ContextCopula().fit(t);model.save(out/'generator.npz')
        a,order=model.draw(args.seed);np.save(out/'synthetic.npy',a,allow_pickle=False);np.save(out/'draw_order.npy',order,allow_pickle=False)
        b,_=ContextCopula.load(out/'generator.npz').draw(args.seed)
        if not np.array_equal(a,b):raise RuntimeError('Generator roundtrip changed output')
        write(out/'GENERATOR.json',{'method':'contextual empirical Gaussian copula','groups':len(model.counts),'singleton_groups':int((model.counts==1).sum()),'context_columns':['wilderness','soil','cover'],'fit_and_draw_seconds':time.monotonic()-start,'covariance':'Gaussian midranks standardized within context, LedoitWolf shrinkage, 1e-6 identity mixture','marginal_inverse':'linear interpolation through empirical sorted training values, nearest integer','all_training_rows_used':len(t),'generated_rows':len(a),'rejected_rows':0,'roundtrip_exact':True,'draw_seed':args.seed,'table_sha256':data_digest(a),'scientific_novelty':False})
        write(out/'PROGRESS.json',{'stage':'generation_completed'})
        quality=benchmark(a,d,t);write(out/'QUALITY.json',quality)
        tree=fit_student(a,d,out/'tree',args.seed);write(out/'PROGRESS.json',{'stage':'tree_completed','tree':tree})
        linear=fit_linear(a,d,out/'linear')
        write(out/'SUMMARY.json',{'seed':args.seed,'tree':tree,'linear':linear,'quality':{k:v for k,v in quality.items() if k not in ['class_conditional_tail_events']},'final_test_scored':False})
        print(json.dumps({'seed':args.seed,'tree':tree['balanced_accuracy'],'linear':linear['balanced_accuracy'],'status':'completed'}),flush=True)
    except Exception as exc:
        write(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--seed',type=int,required=True);main(p.parse_args())
