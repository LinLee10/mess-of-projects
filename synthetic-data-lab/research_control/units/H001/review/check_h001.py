"""Independent artifact acceptance for H001. No import from worker or benchmark."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import lightgbm as lgb
from scipy.stats import ks_2samp


def check(root):
    root=Path(root).resolve();seal=json.loads((root/'DONE.json').read_text())
    for rel,expected in seal['files'].items():
        p=root/rel
        if not p.resolve().is_relative_to(root) or p.is_symlink():raise AssertionError('Bad manifest path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise AssertionError('Hash mismatch '+rel)
    t=np.load(root/'data/train.npy',allow_pickle=False);d=np.load(root/'data/dev.npy',allow_pickle=False)
    with np.load(root/'data/membership.npz',allow_pickle=False) as ids:
        ix=np.concatenate([ids[k] for k in ['train','dev','reserved']]);assert len(ix)==len(np.unique(ix))==581012
    observed={}
    for variant in ['real','raw','pooled_legacy','pooled_keyed','context_keyed']:
        student=root/'real_student' if variant=='real' else root/variant/'student'
        with np.load(student/'predictions.npz',allow_pickle=False) as z:y,p=z['labels'],z['probabilities']
        assert np.array_equal(y,d[:,12]);assert p.shape==(len(y),7);assert np.isfinite(p).all();assert (p>=0).all();assert np.allclose(p.sum(1),1)
        matrix=np.zeros((7,7),dtype=int)
        for c in range(7):matrix[c]=np.bincount(p[y==c].argmax(1),minlength=7)
        row=json.loads((student/'METRICS.json').read_text())['utility']
        assert matrix.tolist()==row['confusion_matrix'];assert int(matrix.trace())==row['correct']
        bal=sum(matrix[c,c]/matrix[c].sum() for c in range(7))/7
        assert abs(bal-row['balanced_accuracy'])<1e-12
        reload=lgb.Booster(model_file=str(student/'model.txt')).predict(d[:,:12],num_threads=2)
        assert np.array_equal(reload,p)
        if variant!='real':
            a=np.load(root/variant/'synthetic.npy',allow_pickle=False);q=json.loads((root/variant/'QUALITY.json').read_text())
            assert len(a)==len(t);assert np.array_equal(np.bincount(a[:,12],minlength=7),np.bincount(t[:,12],minlength=7))
            ks=[float(ks_2samp(a[:,j],d[:,j],method='asymp').statistic) for j in range(10)]
            assert max(abs(x-y) for x,y in zip(ks,q['numeric_ks_by_feature']))<1e-12
            assert abs(sum(ks)/10-q['mean_numeric_ks'])<1e-12
        observed[variant]={'correct':int(matrix.trace()),'n':len(y),'balanced_accuracy':float(bal),'reload_exact':True}
    controls=json.loads((root/'ORDER_CONTROLS.json').read_text())
    assert controls['keyed']['rows_changed_by_permutation']==0
    assert controls['midrank']['rows_changed_by_permutation']==0
    assert controls['legacy']['rows_changed_by_permutation']>0
    return {'status':'passed','verified_files':len(seal['files']),'students':observed,'final_test_scored':False,'independence':'Same assistant, separate process, no worker/benchmark imports','scope':'Manifest, split disjointness, marginals, confusion arithmetic, class counts and reload; not universal fidelity'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.run.resolve()):raise ValueError('Use new output outside sealed run')
    r=check(a.run);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
