"""Descriptive reporting of frozen outcomes; never changes or selects training."""
from __future__ import annotations
from collections import defaultdict
import csv,json
from pathlib import Path
import numpy as np
from .common import verify_seal,write_json

METRICS=('balanced_accuracy','accuracy','macro_f1','log_loss')
ABLATIONS=(('ctgan_unpacked','ctgan','Remove packing'),('ctgan_one_mode','ctgan','Replace mixture with one mode'),
    ('ddpm_unconditional','ddpm','Remove class conditioning'),('ddpm_standardized','ddpm','Replace quantiles with standard scaling'),
    ('mixture_utility','mixture_uniform','Add utility weights'),('mixture_support','mixture_uniform','Add support gate'),
    ('mixture_utility_support','mixture_uniform','Add weights and gate'),
    ('mixture_utility_support','mixture_support','Add weights after gate'))


def describe(values):
    return {'mean':float(np.mean(values)),'sd_across_split_means':float(np.std(values,ddof=1)),
            'min_split_mean':float(np.min(values)),'max_split_mean':float(np.max(values)),
            'split_means':[float(v) for v in values], 'independent_population_experiments':False}


def analyze(root,output):
    root=Path(root);out=Path(output)
    frozen=verify_seal(root);verify_seal(root/'final_evaluation')
    scores=json.loads((root/'final_evaluation/scores.json').read_text())
    if scores.get('frozen_manifest_sha256')!=frozen['manifest_sha256']:
        raise ValueError('Evaluation provenance does not match the frozen candidate manifest')
    if out.exists():raise ValueError('Analysis output already exists; overwrite refused')
    out.mkdir(parents=True)
    rows=scores['rows']
    groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['student'],row['method'],row['regime'])].append(row)
    summary=[]
    for key,items in sorted(groups.items()):
        result=dict(zip(('dataset','student','method','regime'),key));result['scored_conditions']=len(items)
        for metric in METRICS:
            values=[np.mean([r['test'][metric] for r in items if r['split_seed']==s]) for s in sorted({r['split_seed'] for r in items})]
            result[metric]=describe(values)
        summary.append(result)
    deltas=[]
    for dataset in sorted({r['dataset'] for r in rows}):
      for student in ('mlp','logistic'):
       for regime in ('synthetic_only','augment'):
        for left,right,title in ABLATIONS:
            a=groups[(dataset,student,left,regime)];b=groups[(dataset,student,right,regime)]
            lookup={(r['split_seed'],r['generation_seed'],r['student_seed']):r for r in b}
            paired=[(r['split_seed'],r['test']['balanced_accuracy']-lookup[(r['split_seed'],r['generation_seed'],r['student_seed'])]['test']['balanced_accuracy']) for r in a]
            vals=[100*np.mean([value for seed,value in paired if seed==s]) for s in sorted({seed for seed,_ in paired})]
            deltas.append({'dataset':dataset,'student':student,'regime':regime,'left':left,'right':right,'change':title,
                'balanced_accuracy_delta_percentage_points':describe(vals),'positive_split_count':int(sum(v>0 for v in vals)),
                'negative_split_count':int(sum(v<0 for v in vals))})
    interactions=[]
    for dataset in sorted({r['dataset'] for r in rows}):
      for regime in ('synthetic_only','augment'):
        g={m:groups[(dataset,'mlp',m,regime)] for m in ('mixture_uniform','mixture_utility','mixture_support','mixture_utility_support')}
        v=[]
        for s in (17,29,41):
            mean=lambda m:np.mean([r['test']['balanced_accuracy'] for r in g[m] if r['split_seed']==s])
            v.append(100*(mean('mixture_utility_support')-mean('mixture_utility')-mean('mixture_support')+mean('mixture_uniform')))
        interactions.append({'dataset':dataset,'student':'mlp','regime':regime,'definition':'both - utility - support + uniform',
            'percentage_points':describe(v),'interpretation':'Descriptive factorial interaction, not a population causal estimate'})
    selected=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        chosen=[r for r in rows if r['dataset']==dataset and r['student']=='mlp' and r['selected_by_development']]
        selected.append({'dataset':dataset,'selection':'Per replicate development rule fixed before test',
            'balanced_accuracy':describe([np.mean([r['test']['balanced_accuracy'] for r in chosen if r['split_seed']==s]) for s in (17,29,41)]),
            'chosen_conditions':[{'split':r['split_seed'],'generation':r['generation_seed'],'method':r['method'],'regime':r['regime']} for r in chosen if r['student_seed']==7]})
    audit_groups=defaultdict(list);neural=[];probes=[]
    for path in (root/'runs').glob('*/split_*/generation_*/generators/*/audit.json'):
        a=json.loads(path.read_text());dataset=path.parts[-6];method=path.parent.name
        audit_groups[(dataset,method)].append(a)
        if a.get('initial_sha256'):neural.append(a)
    costs=[]
    for (dataset,method),items in sorted(audit_groups.items()):
        d={'dataset':dataset,'method':method,'replicates':len(items)}
        if 'total_fit_and_sample_seconds' in items[0]:d['mean_component_fit_and_sample_seconds']=float(np.mean([a['total_fit_and_sample_seconds'] for a in items]))
        if 'sampling' in items[0]:
            d['mean_generated_proposals']=float(np.mean([a['sampling']['generated'] for a in items]));d['mean_retained_fraction']=float(np.mean([a['sampling']['retained_fraction'] for a in items]))
        for key in ('marginal_wasserstein_train_standardized','correlation_mean_absolute_error_train','exact_training_copy_rate','exact_internal_duplicate_rate','negative_feature_row_fraction'):
            d[key]=float(np.mean([a['diagnostics'][key] for a in items]))
        costs.append(d)
    for p in (root/'runs').glob('*/split_*/generation_*/probes/*/*/model.json'):probes.append(json.loads(p.read_text())['audit'])
    primary=[json.loads((root/r['path']/'model.json').read_text())['audit'] for r in rows if r['student']=='mlp']
    fits=neural+probes+primary
    if not all(a['initial_sha256']!=a['final_sha256'] and np.isfinite(a['losses']).all() and a['completed_steps']>0 for a in fits):
        raise ValueError('A neural fit lacks finite losses, completed steps or changed weights')
    # This report is specific to the declared pilot, including its expected fit counts.
    if (len(primary),len(probes),len(neural))!=(696,96,84):
        raise ValueError('Analysis requires the declared full pilot; a custom protocol needs a separate analysis contract')
    write_json(out/'summary.json',{'groups':summary,'ablations':deltas,'factorial_interactions':interactions,'development_selected_policy':selected,
        'cost_and_diagnostics':costs,'audit':{'neural_generator_fits':len(neural),'primary_mlp_fits':len(primary),'probe_mlp_fits':len(probes),
            'secondary_logistic_fits':sum(r['student']=='logistic' for r in rows),'all_neural_fits_have_finite_losses_and_changed_weights':True,
            'training_elapsed_seconds':json.loads((root/'INDEX.json').read_text())['training_elapsed_seconds'],
            'total_neural_optimization_steps':sum(a['completed_steps'] for a in fits),
            'warning':'Neural steps count learning iterations, not generator plus critic optimizer updates or FLOPs'},
        'uncertainty':'Three overlapping repeated splits; SD across split means is descriptive, not a confidence interval. No universal or statistically confirmed winner.',
        'selection_boundary':'Model exports use only the preregistered Wine split 17 generation 101 development rule, never this aggregate test table.'})
    with (out/'all_conditions.csv').open('w',newline='') as f:
        keys=['dataset','split_seed','generation_seed','student','student_seed','method','regime','selected_by_development',*METRICS]
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in keys if k not in METRICS},**{k:r['test'][k] for k in METRICS}})
    with (out/'mean_results.csv').open('w',newline='') as f:
        keys=['dataset','student','method','regime','balanced_accuracy_percent','split_sd_percentage_points','accuracy_percent','macro_f1','log_loss']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in summary:w.writerow({**{k:r[k] for k in ('dataset','student','method','regime')},
            'balanced_accuracy_percent':100*r['balanced_accuracy']['mean'],'split_sd_percentage_points':100*r['balanced_accuracy']['sd_across_split_means'],
            'accuracy_percent':100*r['accuracy']['mean'],'macro_f1':r['macro_f1']['mean'],'log_loss':r['log_loss']['mean']})
    return {'groups':len(summary),'paired_ablations':len(deltas),'scored_conditions':len(rows)}
