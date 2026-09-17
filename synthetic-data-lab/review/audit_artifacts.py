"""Independent artifact arithmetic and semantic replay comparison.

Does not import synthlab_next or sklearn metrics. This is separate verification
code by the same assistant, not an independent research group endorsement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def check_seal(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'FROZEN.json').read_text())
    expected=manifest.pop('manifest_sha256')
    actual=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(actual==expected,'Modified seal')
    for name,expected_file in manifest['files'].items():
        path=root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root),'Unsafe seal member')
        require(digest_file(path)==expected_file,'Changed artifact: '+name)
    return expected,len(manifest['files'])


def independent_predict(path,x,student):
    m=json.loads((path/'model.json').read_text())
    standardized=(x-np.asarray(m['mean']))/np.asarray(m['scale'])
    if student=='mlp':
        state=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        z=torch.tensor(standardized,dtype=torch.float32)
        with torch.no_grad():
            for i in (0,2,4):
                z=torch.nn.functional.linear(z,state[f'layers.{i}.weight'],state[f'layers.{i}.bias'])
                if i!=4:
                    z=torch.relu(z)
            return torch.softmax(z,dim=1).numpy().astype(float)
    logits=standardized@np.asarray(m['coef']).T+np.asarray(m['intercept'])
    if logits.shape[1]==1:
        logits=np.column_stack((np.zeros(len(x)),logits[:,0]))
    logits-=logits.max(axis=1,keepdims=True)
    p=np.exp(logits)
    return p/p.sum(axis=1,keepdims=True)


def metric_oracle(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(p,1e-12,1);p=p/p.sum(1,keepdims=True)
    pred=p.argmax(1);k=p.shape[1]
    cm=np.zeros((k,k),dtype=np.int64)
    for truth,guess in zip(y,pred):
        cm[truth,guess]+=1
    recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(k),where=cm.sum(1)>0)
    denom=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),denom,out=np.zeros(k),where=denom>0)
    return {'accuracy':float(cm.trace()/len(y)),
            'balanced_accuracy':float(recall[cm.sum(1)>0].mean()),
            'macro_f1':float(f1.mean()),
            'log_loss':float(-np.log(p[np.arange(len(y)),y]).mean()),
            'confusion_matrix':cm.tolist()}


def audit(root):
    root=Path(root).resolve();torch.set_num_threads(1)
    seal_hash,candidate_count=check_seal(root)
    _,evaluation_count=check_seal(root/'final_evaluation')
    scores=json.loads((root/'final_evaluation/scores.json').read_text())
    require(scores['frozen_manifest_sha256']==seal_hash,'Evaluation linked to another candidate seal')
    rows=scores['rows'];require(len(rows)==1044,'Unexpected full pilot condition count')
    errors={key:0. for key in ('accuracy','balanced_accuracy','macro_f1','log_loss')}
    exact_predictions=0;maximum_prediction_error=0.;changed_labels=0
    for row in rows:
        split=root/'splits'/row['dataset']/str(row['split_seed'])
        with np.load(split/'test.npz',allow_pickle=False) as test,np.load(root/row['prediction_path'],allow_pickle=False) as saved:
            require(np.array_equal(test['ids'],saved['ids']),'Prediction ID mismatch')
            require(np.array_equal(test['y'],saved['labels']),'Prediction label mismatch')
            predicted=independent_predict(root/row['path'],test['x'],row['student'])
            exact_predictions+=int(np.array_equal(predicted,saved['probabilities']))
            maximum_prediction_error=max(maximum_prediction_error,float(np.max(abs(predicted-saved['probabilities']))))
            changed_labels+=int(np.sum(predicted.argmax(1)!=saved['probabilities'].argmax(1)))
            require(np.allclose(predicted,saved['probabilities'],rtol=1e-5,atol=1e-6),'Material inference mismatch: '+row['path'])
            # Test reported metric arithmetic against the actual saved probabilities.
            # Bitwise prediction equality is recorded separately, never inferred.
            metrics=metric_oracle(test['y'],saved['probabilities'])
            for key in errors:
                error=abs(metrics[key]-row['test'][key]);errors[key]=max(errors[key],error)
                require(error<1e-12,'Metric mismatch: '+key)
            require(metrics['confusion_matrix']==row['test']['confusion_matrix'],'Confusion mismatch')
    for split in (root/'splits').glob('*/*'):
        ids=[]
        for part in ('train','dev','test','unused'):
            with np.load(split/f'{part}.npz',allow_pickle=False) as data:
                ids.append(set(data['ids'].tolist()))
        require(all(not ids[i]&ids[j] for i in range(4) for j in range(i+1,4)),'Overlapping partition IDs')
    categories={'generator':0,'primary':0,'probe':0,'logistic':0}
    for path in (root/'runs').rglob('model.json'):
        m=json.loads(path.read_text());a=m.get('audit',{})
        if 'students' in path.parts and path.parent.name=='logistic':
            categories['logistic']+=1
        if 'initial_sha256' not in a:
            continue
        require(a['initial_sha256']!=a['final_sha256'],'Unchanged neural weights')
        require(a['completed_steps']>0 and np.isfinite(a['losses']).all(),'Invalid training audit')
        categories['probe' if 'probes' in path.parts else 'primary' if 'students' in path.parts else 'generator']+=1
    require(categories=={'generator':84,'primary':696,'probe':96,'logistic':348},'Fit count mismatch')
    return {'candidate_manifest_sha256':seal_hash,'sealed_candidate_files':candidate_count,
            'sealed_evaluation_files':evaluation_count,'independent_prediction_recomputations':len(rows),
            'independent_metric_checks':len(rows)*4,'maximum_metric_errors':errors,
            'exact_prediction_arrays':exact_predictions,'maximum_reload_probability_error':maximum_prediction_error,
            'changed_reload_class_predictions':changed_labels,
            'inference_comparison_tolerances':{'rtol':1e-5,'atol':1e-6},
            'partition_ID_checks':'passed','neural_fit_counts':categories,
            'reviewer_independence':'Same assistant, separate verifier code and process; no external author endorsement'}


def compare_replay(original,replay):
    """Quantify, rather than conceal, failed bitwise replay acceptance."""
    original,replay=Path(original),Path(replay)
    from collections import defaultdict
    states=defaultdict(lambda:{'total':0,'exact':0,'maximum_absolute_tensor_difference':0.})
    for old in (original/'runs').rglob('weights.pt'):
        new=replay/old.relative_to(original)
        a=torch.load(old,map_location='cpu',weights_only=True);b=torch.load(new,map_location='cpu',weights_only=True)
        require(a.keys()==b.keys(),'Model tensor keys differ')
        kind='generator' if 'generators' in old.parts else 'probe' if 'probes' in old.parts else 'student'
        states[kind]['total']+=1;states[kind]['exact']+=int(all(torch.equal(a[k],b[k]) for k in a))
        for key in a:
            require(a[key].shape==b[key].shape,'Model tensor shape mismatch')
            diff=float((a[key].double()-b[key].double()).abs().max())
            states[kind]['maximum_absolute_tensor_difference']=max(states[kind]['maximum_absolute_tensor_difference'],diff)
    arrays=defaultdict(lambda:{'total_archives':0,'exact_archives':0,'maximum_absolute_array_difference':0.})
    for old in original.rglob('*.npz'):
        new=replay/old.relative_to(original)
        kind='splits' if 'splits' in old.parts else 'test_predictions' if 'final_evaluation' in old.parts else 'training_or_development'
        with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
            require(set(a.files)==set(b.files),'Array keys differ')
            exact=True
            for key in a.files:
                require(a[key].shape==b[key].shape,'Array shape mismatch')
                exact=exact and np.array_equal(a[key],b[key])
                if a[key].size:
                    diff=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float))))
                    arrays[kind]['maximum_absolute_array_difference']=max(arrays[kind]['maximum_absolute_array_difference'],diff)
            arrays[kind]['total_archives']+=1;arrays[kind]['exact_archives']+=int(exact)
            if kind=='splits':require(exact,'Partition or source data changed')
    selections=[]
    for old in (original/'runs').rglob('development_selection.json'):
        a=json.loads(old.read_text());b=json.loads((replay/old.relative_to(original)).read_text())
        aw,bw=a['winner'],b['winner']
        selections.append({'replicate':str(old.parent.relative_to(original)),
            'original_winner':aw,'replay_winner':bw,
            'same_selected_condition':(aw['method'],aw['regime'])==(bw['method'],bw['regime'])})
    a=json.loads((original/'final_evaluation/scores.json').read_text())['rows']
    b={r['path']:r for r in json.loads((replay/'final_evaluation/scores.json').read_text())['rows']}
    require(len(a)==len(b),'Condition count changed')
    errors=defaultdict(float);changes=[];group_deltas=defaultdict(list);changed_labels=0
    for old in a:
        new=b[old['path']]
        delta=new['test']['balanced_accuracy']-old['test']['balanced_accuracy']
        key='|'.join((old['dataset'],old['student'],old['method'],old['regime']))
        group_deltas[key].append(delta)
        for metric in ('accuracy','balanced_accuracy','macro_f1','log_loss'):
            errors[metric]=max(errors[metric],abs(new['test'][metric]-old['test'][metric]))
        with np.load(original/old['prediction_path'],allow_pickle=False) as x,np.load(replay/new['prediction_path'],allow_pickle=False) as y:
            changed_labels+=int(np.sum(x['probabilities'].argmax(1)!=y['probabilities'].argmax(1)))
        if delta:
            changes.append({'path':old['path'],'original_balanced_accuracy':old['test']['balanced_accuracy'],
                'replay_balanced_accuracy':new['test']['balanced_accuracy'],'delta_percentage_points':100*delta})
    exact=all(v['total']==v['exact'] for v in states.values()) and all(v['total_archives']==v['exact_archives'] for v in arrays.values())
    return {'bitwise_replay_passed':exact,'model_states':dict(states),'numpy_archives':dict(arrays),
            'development_selections':selections,'same_development_winner_count':sum(v['same_selected_condition'] for v in selections),
            'evaluated_condition_count':len(a),'conditions_with_changed_balanced_accuracy':len(changes),
            'changed_class_predictions_across_conditions':changed_labels,'maximum_scalar_metric_differences':dict(errors),
            'group_mean_balanced_accuracy_deltas_percentage_points':{k:100*float(np.mean(v)) for k,v in sorted(group_deltas.items())},
            'changed_conditions':changes,'timestamps_and_duration_fields_compared':False,
            'scope':'Fresh execution of the same protocol, not new independent statistical evidence. Failed exact replay remains a failed acceptance criterion; no cause is asserted.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'Audit output already exists')
    report={'status':'passed','original':audit(args.run)}
    if args.replay:
        report['replay']=audit(args.replay);report['comparison']=compare_replay(args.run,args.replay)
        if not report['comparison']['bitwise_replay_passed']:
            report['status']='artifact_checks_passed_bitwise_replay_failed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
