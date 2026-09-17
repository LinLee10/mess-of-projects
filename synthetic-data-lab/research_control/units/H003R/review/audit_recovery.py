"""Read-only recovery audit, with no production scoring or generator imports."""
from pathlib import Path
import hashlib,json,sys,os
import numpy as np
import lightgbm as lgb
ROOT=Path(os.environ.get('RECOVERED_ROOT','/mnt/data/recovered'))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def score(y,p):
    require(y.ndim==1 and p.shape==(len(y),7),'shape')
    require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'bounds')
    require(np.allclose(p.sum(1),1,rtol=0,atol=1e-10),'normalization')
    require(np.isin(y,np.arange(7)).all(),'labels')
    cm=np.bincount(y*7+p.argmax(1),minlength=49).reshape(7,7)
    require((cm.sum(1)>0).all(),'missing class');r=cm.diagonal()/cm.sum(1)
    return dict(n=len(y),correct=int(cm.trace()),accuracy=float(cm.trace()/len(y)),balanced_accuracy=float(r.mean()),class_recall=r.tolist(),confusion_matrix=cm.tolist())
def compare(a,b):
    for k in ['n','correct','confusion_matrix']:require(a[k]==b[k],'metric '+k)
    for k in ['accuracy','balanced_accuracy','class_recall']:require(np.allclose(a[k],b[k],rtol=0,atol=1e-13),'metric '+k)
def check_model(f,d):
    with np.load(f/'predictions.npz',allow_pickle=False) as z:y,p=z['labels'],z['probabilities']
    require(np.array_equal(y,d[:,12]),'wrong development labels');s=score(y,p)
    compare(s,json.loads((f/'METRICS.json').read_text())['utility'])
    if (f/'model.txt').exists():q=lgb.Booster(model_file=str(f/'model.txt')).predict(d[:,:12],num_threads=2)
    else:
        m=json.loads((f/'model.json').read_text());require(m['converged'],'nonconverged')
        x=np.zeros((len(d),54));x[:,:10]=(d[:,:10]-m['mean'])/m['scale']
        x[np.arange(len(d)),10+d[:,10]]=1;x[np.arange(len(d)),14+d[:,11]]=1
        z=x@np.array(m['coef']).T+m['intercept'];z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
    require(np.allclose(p,q,rtol=1e-9,atol=1e-10),'inference mismatch')
    require(np.array_equal(p.argmax(1),q.argmax(1)),'prediction classes')
    s['probability_max_error']=float(np.abs(p-q).max());return s
def main(names):
    d=np.load(ROOT/'h001-seed17/run/data/dev.npy',allow_pickle=False);t=np.load(ROOT/'h001-seed17/run/data/train.npy',allow_pickle=False)
    require(t.shape==(348563,13) and d.shape==(116314,13),'data shapes')
    tc,ti,tn=np.unique(t[:,10:],axis=0,return_inverse=True,return_counts=True)
    sorted_training=[np.sort(t[ti==i,:10],axis=0) for i in range(len(tc))]
    report={'final_test_scored':False,'authorship':'Same assistant, separate read-only implementation and process','results':{}}
    for name in names:
        root=ROOT/name;manifest=json.loads((root/'ARTIFACT_SHA256.json').read_text())
        for rel,h in manifest.items():require(hashlib.file_digest((root/rel).open('rb'),'sha256').hexdigest()==h,'digest '+rel)
        methods=['gaussian_rank','beta_rank','beta_wide_rank','independent_rank'] if name.startswith('h002') else ['neighbor_raw','neighbor_rank','adaptive_beta']
        e={'verified_files':len(manifest),'methods':{}}
        for method in methods:
            folder=root/'run'/method;s={f:check_model(folder/f,d) for f in ['tree','linear']}
            table=np.load(folder/'synthetic.npy',allow_pickle=False);require(table.shape==t.shape,'table shape')
            sc,si,sn=np.unique(table[:,10:],axis=0,return_inverse=True,return_counts=True)
            require(np.array_equal(tc,sc) and np.array_equal(tn,sn),'context counts')
            if method!='neighbor_raw':
                for i in range(len(tc)):require(np.array_equal(np.sort(table[si==i,:10],axis=0),sorted_training[i]),'conditional marginals')
            if (folder/'PROXIMITY.json').exists():
                with np.load(folder/'proximity.npz',allow_pickle=False) as z:sd=z['distance']
                with np.load(root/'run/real_development_distances.npz',allow_pickle=False) as z:rd=z['distance']
                pr=json.loads((folder/'PROXIMITY.json').read_text());n=int((sd<np.quantile(rd,.05)).sum())
                require(n==pr['strictly_below_real_q05_count'],'proximity count');s['proximity']=pr
            e['methods'][method]=s;print(name,method,s['tree']['balanced_accuracy'],flush=True)
        report['results'][name]=e
    report['real_reference_seed17']=check_model(ROOT/'h001-seed17/run/real_student',d)
    report['status']='passed'
    out=Path(os.environ.get('AUDIT_OUTPUT_ROOT','/mnt/data/continuation/results'))/('audit_'+'_'.join(names)+'.json');out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1:])
