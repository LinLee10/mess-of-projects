"""Read only checker. No imports from either implementation under review.

Verifies stored arithmetic at 1e-10 and neural checkpoint inference separately.
Same assistant authorship; separate code and process, not independent authorship.
"""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def need(value,msg):
    if not value:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):
    a=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def sh(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def check_seal(root):
    m=read(root/'FROZEN.json');stored=m.pop('manifest_sha256')
    need(hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==stored,'Seal mismatch')
    need(bool(m['files']),'Empty seal')
    for rel,d in m['files'].items():
        p=root/rel;need(p.resolve().is_relative_to(root.resolve()) and p.is_file() and not p.is_symlink(),'Unsafe artifact')
        need(digest(p)==d,'File changed: '+rel)
    return stored,len(m['files'])
def check_scores(y,p,reported):
    need(y.ndim==1 and len(y)>0 and np.issubdtype(y.dtype,np.integer),'Invalid labels')
    need(p.ndim==2 and len(p)==len(y) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,rtol=0,atol=1e-5),'Invalid probabilities')
    need(p.shape[1]>=2 and (y>=0).all() and (y<p.shape[1]).all(),'Unknown labels')
    pred=np.argmax(p,1);k=p.shape[1];cm=[[0]*k for _ in range(k)]
    for t,g in zip(y,pred):cm[int(t)][int(g)]+=1
    recall=[cm[c][c]/sum(cm[c]) for c in range(k) if sum(cm[c])]
    f1=[2*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)) if sum(cm[c])+sum(row[c] for row in cm) else 0 for c in range(k)]
    q=np.clip(p.astype(float),1e-12,1);q=q/q.sum(1,keepdims=True)
    ll=-sum(np.log(q[i,int(v)]) for i,v in enumerate(y))/len(y)
    correct=sum(int(t==g) for t,g in zip(y,pred))
    expected={'n':len(y),'correct':correct,'accuracy':correct/len(y),'balanced_accuracy':sum(recall)/len(recall),'macro_f1':sum(f1)/len(f1),'log_loss':ll}
    for key,v in expected.items():need(abs(float(reported[key])-v)<=1e-10,'Metric mismatch: '+key)
    need(reported['confusion_matrix']==cm,'Confusion matrix mismatch')
    return expected

def audit(root):
    root=Path(root);torch.set_num_threads(1);ch,nc=check_seal(root);eh,ne=check_seal(root/'evaluation')
    c=read(root/'CANDIDATES.json');e=read(root/'evaluation/scores.json');need(e['candidate_manifest']==ch,'Wrong experiment provenance')
    need(len(c['rows'])==len(e['rows']),'Candidate omitted')
    sources={};sourcechecks=0
    for d in (root/'data').iterdir():
        meta=read(d/'source.json');parts={};indices=[]
        x=np.empty((meta['source_rows'],len(meta['features'])));y=np.empty(meta['source_rows'],dtype=np.int64);groups={}
        for part in ['train','dev','test']:
            with np.load(d/(part+'.npz'),allow_pickle=False) as z:parts[part]={k:z[k].copy() for k in z.files}
            z=parts[part];indices.extend(z['ids'].tolist());x[z['ids']]=z['x'];y[z['ids']]=z['y']
            for row in z['x']:
                a=np.array(row,dtype='<f8');a[a==0]=0;b=a.tobytes()
                need(b not in groups or groups[b]==part,'Exact feature leakage');groups[b]=part
        need(len(indices)==len(set(indices))==meta['source_rows'],'Rows missing or split overlap')
        need(ah(x)==meta['source_x_sha256'] and ah(y)==meta['source_y_sha256'],'Source lineage changed')
        sources[meta['dataset']]=parts;sourcechecks+=1
    exact=0;maxerr=0.;predchecks=0;coverage=0;steps={};seen=set()
    for original,row in zip(c['rows'],e['rows']):
        need(original['path']==row['path'] and row['path'] not in seen,'Candidate identity mismatch');seen.add(row['path'])
        with np.load(root/row['prediction_path'],allow_pickle=False) as z:p=z['probabilities'];y=z['labels'];ids=z['ids']
        test=sources[row['test_domain']]['test'];need(np.array_equal(y,test['y']) and np.array_equal(ids,test['ids']),'Wrong heldout target')
        check_scores(y,p,row['test']);folder=root/row['path']
        with np.load(folder/'dev.npz',allow_pickle=False) as z:
            need(np.array_equal(z['labels'],sources[row['test_domain']]['dev']['y']),'Wrong development target')
            check_scores(z['labels'],z['probabilities'],row['dev'])
        if row['student']=='mlp':
            meta=read(folder/'model.json');a=meta['audit'];weights=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
            need(sh(weights)==a['final_sha256'] and a['final_sha256']!=a['initial_sha256'],'Weight identity')
            need(a['minimum_row_exposures']>=1 and a['unique_training_rows_seen']==row['training_rows'],'Incomplete training coverage')
            with np.load(root/row['data_path'],allow_pickle=False) as z:need(ah(z['x'])==a['training_x_sha256'] and ah(z['y'])==a['training_y_sha256'],'Training data lineage')
            if row['real_path']:
                with np.load(root/row['real_path'],allow_pickle=False) as z:need(ah(z['x'])==a['real_x_sha256'] and ah(z['y'])==a['real_y_sha256'] and a['real_unique_rows_seen']==len(z['y']),'Real anchor lineage')
            with torch.no_grad():
                h=torch.tensor((test['x']-meta['mean'])/meta['scale'],dtype=torch.float32)
                for layer in [0,2,4]:
                    h=F.linear(h,weights[f'layers.{layer}.weight'],weights[f'layers.{layer}.bias'])
                    if layer<4:h=F.relu(h)
                replay=F.softmax(h,dim=1).numpy()
            need(np.array_equal(p.argmax(1),replay.argmax(1)),'Checkpoint decisions changed')
            need(np.allclose(p,replay,rtol=1e-5,atol=1e-6),'Checkpoint probabilities changed')
            maxerr=max(maxerr,float(np.abs(p-replay).max()));exact+=int(np.array_equal(p,replay));predchecks+=1;coverage+=1
        else:
            # Own sealed classifier only. Do not use this loader for untrusted external pickle files.
            model=pickle.loads((folder/'model.pkl').read_bytes());replay=model.predict_proba(test['x'])
            need(np.allclose(p,replay,rtol=1e-5,atol=1e-6) and np.array_equal(p.argmax(1),replay.argmax(1)),'Tree reload mismatch');predchecks+=1
    generatorchecks=0
    for rel in c['generator_paths']:
        a=read(root/rel/'audit.json')['fit']
        if (root/rel/'weights.pt').exists():
            need(a['minimum_row_exposures']>=1,'Generator omitted real rows')
            need(a['unique_training_rows_seen']==a['training_records'],'Generator coverage wrong')
            need(a['completed_steps']==len(a['losses']),'Generator stopped early')
            weights=torch.load(root/rel/'weights.pt',weights_only=True,map_location='cpu');need(sh(weights)==a['final_sha256'],'Generator weights changed');generatorchecks+=1
    return {'status':'passed','stored_conditions_checked':len(e['rows']),'source_partitions_checked':sourcechecks,'generator_coverage_checks':generatorchecks,'neural_student_coverage_checks':coverage,'checkpoint_reloads':predchecks,'neural_arrays_exact':exact,'maximum_neural_reload_difference':maxerr,'candidate_manifest':ch,'evaluation_manifest':eh,'candidate_files_checked':nc,'evaluation_files_checked':ne,'production_modules_imported':any(k.startswith('synthlab_next') or k=='fullscale' for k in sys.modules),'independent_author':False,'limits':['Artifact consistency is not proof of distributional adequacy','Inference reload is not complete retraining','Trees reload through sklearn; neural inference is separately reconstructed','All explicit failed methods remain in the candidate index']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists() and not a.output.resolve().is_relative_to(a.root.resolve()),'Audit output must be new and outside experiment')
    value=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
