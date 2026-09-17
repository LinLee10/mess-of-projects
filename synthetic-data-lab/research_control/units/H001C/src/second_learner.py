"""H001C: prespecified linear student challenge of fixed, recovered H001 tables."""
from __future__ import annotations
import argparse, hashlib, json, os, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning


def validate(a):
    if a.ndim!=2 or a.shape[1]!=13 or a.dtype.kind not in 'iu':raise ValueError('Invalid compact input')
    for j,k in [(10,4),(11,40),(12,7)]:
        if ((a[:,j]<0)|(a[:,j]>=k)).any():raise ValueError('Invalid categorical value')


def encode(a, mean, scale):
    validate(a)
    return np.column_stack(((a[:,:10]-mean)/scale,np.eye(4)[a[:,10]],np.eye(40)[a[:,11]]))


def summarize(y,p):
    pred=p.argmax(1);cm=np.zeros((7,7),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    return {'n':len(y),'correct':int(np.trace(cm)),'accuracy':float(np.mean(y==pred)),
      'balanced_accuracy':float(np.mean(cm.diagonal()/cm.sum(1))),
      'class_recall':(cm.diagonal()/cm.sum(1)).tolist(),'confusion_matrix':cm.tolist()}


def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def run(root,out):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists() or out.is_relative_to(root):raise ValueError('Use new output outside immutable input')
    out.mkdir(parents=True)
    manifest=json.loads((root/'ARTIFACT_SHA256.json').read_text())
    for rel,h in manifest.items():
        path=root/rel
        if not path.resolve().is_relative_to(root) or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('Input artifact failed checksum')
    r=root/'run';train=np.load(r/'data/train.npy',allow_pickle=False);dev=np.load(r/'data/dev.npy',allow_pickle=False)
    if (len(train),len(dev))!=(348563,116314):raise ValueError('Unexpected input dimensions')
    write(out/'START.json',{'source_commit':os.environ.get('GITHUB_SHA'),'input_manifest_sha256':hashlib.sha256((root/'ARTIFACT_SHA256.json').read_bytes()).hexdigest(),'input_source':json.loads((r/'START.json').read_text()),'full_training_rows':len(train),'development_rows':len(dev),'final_test_scored':False})
    results={}
    for name in ['real','raw','pooled_legacy','pooled_keyed','context_keyed']:
        dest=out/name;dest.mkdir();t0=time.monotonic()
        a=train if name=='real' else np.load(r/name/'synthetic.npy',allow_pickle=False)
        validate(a);mean=a[:,:10].mean(0);scale=np.maximum(a[:,:10].std(0),1e-8)
        x=encode(a,mean,scale);dx=encode(dev,mean,scale)
        model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,tol=1e-5)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');model.fit(x,a[:,12])
        convergence=not any(issubclass(w.category,ConvergenceWarning) for w in ws)
        p=model.predict_proba(dx);score=summarize(dev[:,12],p)
        np.savez_compressed(dest/'predictions.npz',labels=dev[:,12],probabilities=p)
        write(dest/'model.json',{'mean':mean.tolist(),'scale':scale.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),'configuration':{'C':1.,'solver':'lbfgs','max_iter':1000,'tol':1e-5},'iterations':model.n_iter_.tolist(),'converged':convergence,'warnings':[str(w.message) for w in ws]})
        results[name]={'utility':score,'training_seconds':time.monotonic()-t0,'converged':convergence,'training_rows':len(a)}
        write(dest/'RESULT.json',results[name]);write(out/'PROGRESS.json',results)
        print(json.dumps({'variant':name,**results[name]}),flush=True)
        if not convergence:raise RuntimeError('Convergence gate failed; retained artifacts are incomplete')
        del x,dx,model,p
    write(out/'SUMMARY.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
