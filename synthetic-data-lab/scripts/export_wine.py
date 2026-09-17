"""Export the predeclared representative Wine models, without test selection."""
from pathlib import Path
import json,shutil
import numpy as np
import torch
from synthlab_next.common import verify_seal,write_json,array_hash,seal
from synthlab_next.inference import sample_csv,predict_csv
from synthlab_next.diagnostics import table_diagnostics

root=Path(__file__).resolve().parents[1];run=root/'results/pilot_v1'
verify_seal(run);verify_seal(run/'final_evaluation');torch.set_num_threads(1)
policy=json.loads((root/'configs/export_policy.json').read_text());rep=run/'runs/wine/split_17/generation_101'
selection=json.loads((rep/'development_selection.json').read_text());overall=selection['winner']
noncopy=next(r for r in selection['ranked_candidates'] if r['regime']=='synthetic_only' and r['method'] not in {'bootstrap','real_only'})
out=root/'models/wine_demo'
if out.exists():raise ValueError('Existing export is immutable')
out.mkdir(parents=True)
meta=json.loads((run/'splits/wine/17/metadata.json').read_text());schema={k:meta[k] for k in ('feature_names','target_names')}
schema['dataset']='wine';schema['use']='Research benchmark only; class is cultivar, not quality'
with np.load(run/'splits/wine/17/train.npz',allow_pickle=False) as a:rx,ry=a['x'],a['y']
for label,choice in [('selected_student',overall),('synthetic_student',noncopy)]:
    src=rep/'students'/choice['method']/choice['regime']/str(policy['exported_student_seed'])
    shutil.copytree(src,out/label);write_json(out/label/'schema.json',schema)

def export_generator(method,destination):
    if not method.startswith('mixture_'):
        shutil.copytree(rep/'generators'/method,destination,
            ignore=shutil.ignore_patterns('pool.npz','audit.json'))
    else:
        destination.mkdir();component_names=['copula','ctgan','tvae','ddpm']
        for name in component_names:
            shutil.copytree(rep/'generators'/name,destination/'components'/name,
                ignore=shutil.ignore_patterns('pool.npz','audit.json'))
        inputs=json.loads((rep/'portfolio_inputs.json').read_text())
        audit=json.loads((rep/'generators'/method/'audit.json').read_text())
        np.savez_compressed(destination/'reference.npz',x=rx,y=ry)
        write_json(destination/'model.json',{'method':'portfolio','variant':method,'n_features':rx.shape[1],
            'n_classes':len(np.unique(ry)),'class_prob':(np.bincount(ry)/len(ry)).tolist(),
            'components':{n:'components/'+n for n in component_names},'weighted':audit['weighted'],'gated':audit['gated'],
            'development_losses':inputs['per_class_development_losses'],'development_counts':inputs['development_class_counts'],
            'reference_x_sha256':array_hash(rx),'reference_y_sha256':array_hash(ry),'audit':audit})
    write_json(destination/'schema.json',schema)

export_generator(noncopy['method'],out/'selected_generator')
export_generator(policy['experimental_portfolio'],out/'experimental_portfolio')
samples=sample_csv(out/'selected_generator',out/'synthetic_wine_1000.csv',policy['sample_count'],policy['sample_seed'])
prediction=predict_csv(out/'synthetic_student',out/'synthetic_wine_1000.csv',out/'synthetic_predictions.csv')
import csv
with (out/'synthetic_wine_1000.csv').open() as f:
    data=list(csv.DictReader(f));sx=np.array([[float(r[k]) for k in schema['feature_names']] for r in data]);sy=np.array([int(r['class_id']) for r in data])
diagnostics=table_diagnostics(rx,ry,sx,sy)
scores=json.loads((run/'final_evaluation/scores.json').read_text())['rows']
check=lambda choice:next(r['test'] for r in scores if r['dataset']=='wine' and r['split_seed']==17 and r['generation_seed']==101 and r['student']=='mlp' and r['student_seed']==7 and r['method']==choice['method'] and r['regime']==choice['regime'])
write_json(out/'EXPORT.json',{'policy':policy,'selected_student':overall,'selected_noncopying_generator':noncopy,
    'selected_student_test':check(overall),'synthetic_student_test':check(noncopy),
    'fresh_sample_diagnostics':diagnostics,'sampling':samples,'prediction':prediction,
    'test_selection':False,'scope':'Single fixed representative split, not an independently replicated production model',
    'privacy':'Reference arrays and learned statistics are retained. No confidentiality guarantee.'})
seal(out,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()])
print(json.dumps({'selected_student':overall,'selected_generator':noncopy,'sample_diagnostics':diagnostics},indent=2))
