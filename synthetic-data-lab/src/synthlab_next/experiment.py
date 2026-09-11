"""Bounded real experiments with an explicit freeze before final evaluation.

The training routine never reads the held out test artifact after partition
creation. Only the separate evaluate entry point loads it, after verifying every
candidate, source snapshot, prediction rule and development decision.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import shutil
import numpy as np
import scipy,sklearn,torch
from .common import write_json,file_hash,canonical_hash,array_hash,seal,verify_seal
from .data import make_split
from .generators import CTGAN,TVAE,TabDDPM,Copula,Interpolation,GeneratorConfig
from .students import StudentConfig,train_student,load_student,classification_metrics,fit_logistic,logistic_predict
from .portfolio import select_portfolio
from .diagnostics import table_diagnostics


def labels_for(y,n,seed):
    """Identical empirical class allocations for every method in one comparison."""
    p=np.bincount(y)/len(y);counts=np.floor(p*n).astype(int)
    residual=n-counts.sum()
    for c in np.argsort(-(p*n-counts),kind='stable')[:residual]:counts[c]+=1
    labels=np.repeat(np.arange(len(p)),counts);np.random.default_rng(seed).shuffle(labels)
    return labels


def subset(x,y,labels,seed):
    rng=np.random.default_rng(seed);out=np.empty((len(labels),x.shape[1]))
    for c in np.unique(labels):
        slots=np.flatnonzero(labels==c);choices=np.flatnonzero(y==c)
        if len(choices)<len(slots):raise ValueError('Insufficient class rows for a matched size subset')
        out[slots]=x[rng.choice(choices,len(slots),replace=False)]
    return out,labels.copy()


def factory(name,config,seed):
    cfg=GeneratorConfig(steps=config['generator_steps'],batch_size=config['generator_batch_size'],
        width=config['generator_width'],latent=config['generator_latent'],diffusion_steps=config['diffusion_steps'])
    if name=='ctgan_unpacked':return CTGAN(replace(cfg,pac=1),seed)
    if name=='ctgan_one_mode':return CTGAN(replace(cfg,max_modes=1),seed)
    if name=='ddpm_unconditional':return TabDDPM(replace(cfg,class_conditioning=False),seed)
    if name=='ddpm_standardized':return TabDDPM(replace(cfg,quantile_transform=False),seed)
    if name in {'ctgan','tvae','ddpm'}:return {'ctgan':CTGAN,'tvae':TVAE,'ddpm':TabDDPM}[name](cfg,seed)
    if name in {'copula','smote'}:return {'copula':Copula,'smote':Interpolation}[name](seed)
    raise ValueError('Unknown generator configuration')


def _log(root,event,**fields):
    row={'time_utc':datetime.now(timezone.utc).isoformat(),'event':event,**fields}
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    print(json.dumps(row),flush=True)


def _run_rep(root,rep,part,config,generation_seed):
    """part contains only training and development arrays, never test arrays."""
    rep.mkdir(parents=True);rx,ry=part['train']['x'],part['train']['y'];dx,dy=part['dev']['x'],part['dev']['y']
    desired=labels_for(ry,config['student_synthetic_size'],generation_seed+31)
    requested=labels_for(ry,config['candidate_pool_size'],generation_seed+37)
    pools={};tables={};audits={};probe_losses=[]
    for method in config['methods']:
        _log(root,'generator_started',replicate=str(rep.relative_to(root)),method=method)
        destination=rep/'generators'/method;destination.mkdir(parents=True)
        start=time.monotonic()
        if method=='bootstrap':
            rng=np.random.default_rng(generation_seed);x=np.empty((len(requested),rx.shape[1]))
            for c in np.unique(ry):
                slots=np.flatnonzero(requested==c);x[slots]=rx[rng.choice(np.flatnonzero(ry==c),len(slots),replace=True)]
            y=requested.copy();audit={'method':'bootstrap','copying_baseline':True,'real_model_training':False,
                'training_x_sha256':array_hash(rx),'training_y_sha256':array_hash(ry),'fit_seconds':0.,
                'sampling':{'generated':len(y),'requested':len(y),'retained_fraction':1.}}
            write_json(destination/'model.json',audit)
        else:
            model=factory(method,config,generation_seed).fit(rx,ry)
            x,y=model.sample_labels(requested,seed=generation_seed+503)
            model.save(destination);audit={**model.audit,'configuration_name':method,'sampling':model.sample_audit}
        np.savez_compressed(destination/'pool.npz',x=x,y=y)
        tx,ty=subset(x,y,desired,generation_seed+907);tables[method]=(tx,ty);pools[method]=(x,y)
        audit['total_fit_and_sample_seconds']=time.monotonic()-start
        audit['pool_x_sha256']=array_hash(x);audit['pool_y_sha256']=array_hash(y)
        audit['diagnostics']=table_diagnostics(rx,ry,tx,ty)
        write_json(destination/'audit.json',audit);audits[method]=audit
        _log(root,'generator_completed',replicate=str(rep.relative_to(root)),method=method,
            seconds=round(audit['total_fit_and_sample_seconds'],3),generated=audit['sampling']['generated'])
    for source in config['portfolio_sources']:
        losses=[]
        for seed in config['student_seeds']:
            probe=train_student(*tables[source],StudentConfig(steps=config['probe_steps'],seed=seed),synthetic=True)
            scores=classification_metrics(dy,probe.predict_proba(dx));losses.append(scores['per_class_loss'])
            probe.save(rep/'probes'/source/str(seed));write_json(rep/'probes'/source/str(seed)/'development.json',scores)
        probe_losses.append(np.mean(losses,axis=0))
    probe_losses=np.array(probe_losses);counts=np.bincount(dy)
    write_json(rep/'portfolio_inputs.json',{'source_order':config['portfolio_sources'],'per_class_development_losses':probe_losses.tolist(),
        'development_class_counts':counts.tolist(),'probe_training_seeds':config['student_seeds'],
        'probe_optimization_steps_per_seed':config['probe_steps'],
        'source_training_cost_seconds':sum(audits[s]['total_fit_and_sample_seconds'] for s in config['portfolio_sources'])})
    for method in config['portfolios']:
        weighted=method in {'mixture_utility','mixture_utility_support'};gated=method in {'mixture_support','mixture_utility_support'}
        x,y,audit=select_portfolio({s:pools[s] for s in config['portfolio_sources']},rx,ry,probe_losses,counts,desired,
            seed=generation_seed+811,weighted=weighted,gated=gated)
        tables[method]=(x,y);audit['diagnostics']=table_diagnostics(rx,ry,x,y);audits[method]=audit
        write_json(rep/'generators'/method/'audit.json',audit)
    for method,(x,y) in tables.items():
        path=rep/'datasets'/method;path.mkdir(parents=True)
        np.savez_compressed(path/'training.npz',x=x,y=y)
        write_json(path/'manifest.json',{'x_sha256':array_hash(x),'y_sha256':array_hash(y),
            'source_real_x_sha256':array_hash(rx),'class_counts':np.bincount(y).tolist(),
            'rows':len(y),'configuration':method,'generation_seed':generation_seed})
    conditions=[('real_only','real_only',rx,ry,None)]
    conditions.extend((name,regime,x,y,(rx,ry) if regime=='augment' else None)
        for name,(x,y) in tables.items() for regime in config['regimes'])
    rows=[]
    for method,regime,x,y,real in conditions:
        for seed in config['student_seeds']:
            path=rep/'students'/method/regime/str(seed)
            trained=train_student(x,y,StudentConfig(steps=config['student_steps'],seed=seed),real=real,synthetic=method!='real_only')
            trained.save(path);p=trained.predict_proba(dx);scores=classification_metrics(dy,p)
            np.savez_compressed(path/'development_predictions.npz',probabilities=p)
            row={'method':method,'regime':regime,'student':'mlp','student_seed':seed,
                'path':str(path.relative_to(root)),'development':scores,'initial_sha256':trained.audit['initial_sha256']}
            rows.append(row)
        path=rep/'students'/method/regime/'logistic';logistic=fit_logistic(x,y,real=real)
        write_json(path/'model.json',logistic);p=logistic_predict(logistic,dx)
        rows.append({'method':method,'regime':regime,'student':'logistic','student_seed':None,
            'path':str(path.relative_to(root)),'development':classification_metrics(dy,p)})
        _log(root,'students_completed',replicate=str(rep.relative_to(root)),method=method,regime=regime)
    # Freeze a candidate by mean development performance across assigned neural seeds.
    ranking=[]
    for method,regime,*_ in conditions:
        matched=[r for r in rows if r['student']=='mlp' and r['method']==method and r['regime']==regime]
        ranking.append({'method':method,'regime':regime,
            'balanced_accuracy':float(np.mean([r['development']['balanced_accuracy'] for r in matched])),
            'log_loss':float(np.mean([r['development']['log_loss'] for r in matched]))})
    ranking.sort(key=lambda r:(-r['balanced_accuracy'],r['log_loss'],r['method'],r['regime']))
    # All methods share precisely the same initial student parameters for a given seed.
    for seed in config['student_seeds']:
        assert len({r['initial_sha256'] for r in rows if r['student']=='mlp' and r['student_seed']==seed})==1
    write_json(rep/'development_selection.json',{'rule':config['selection'],'winner':ranking[0],'ranked_candidates':ranking,
        'selection_uses_test':False,'scope':'One split and generator seed; mean across student seeds'})
    write_json(rep/'index.json',{'rows':rows,'generation_seed':generation_seed,
        'split_path':str(part['path'].relative_to(root)),'selection':ranking[0],'status':'completed'})
    return str((rep/'index.json').relative_to(root))


def train_experiment(config_path,output):
    root=Path(output).resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Output must be new or empty; existing evidence is never overwritten')
    root.mkdir(parents=True,exist_ok=True);config=json.loads(Path(config_path).read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    write_json(root/'protocol.json',config)
    for source in Path(__file__).parent.glob('*.py'):
        dest=root/'code_snapshot'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
    environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'scikit_learn':sklearn.__version__,'torch':torch.__version__,'torch_num_threads':torch.get_num_threads(),
        'gpu_used':False,'pretrained_models_used':False,'network_model_calls':0,
        'executed_track':'continuous tabular data with categorical target','paid_compute':False}
    write_json(root/'environment.json',environment);started=time.monotonic();indices=[]
    try:
        for dataset in config['datasets']:
            for split_seed in config['split_seeds']:
                split=make_split(dataset,split_seed);part_path=root/'splits'/dataset/str(split_seed);part_path.mkdir(parents=True)
                write_json(part_path/'metadata.json',split['metadata'])
                for kind in ('train','dev','test','unused'):np.savez_compressed(part_path/f'{kind}.npz',**split[kind])
                # Do not give test or unused arrays to generator/student training.
                part={'train':split['train'],'dev':split['dev'],'path':part_path};del split
                for generation_seed in config['generation_seeds']:
                    rep=root/'runs'/dataset/f'split_{split_seed}'/f'generation_{generation_seed}'
                    indices.append(_run_rep(root,rep,part,config,generation_seed))
        write_json(root/'INDEX.json',{'replicates':indices,'status':'training_completed','test_evaluated':False,
            'training_elapsed_seconds':time.monotonic()-started})
        paths=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='FROZEN.json']
        frozen=seal(root,paths)
        print(json.dumps({'event':'frozen','manifest_sha256':frozen['manifest_sha256'],'replicates':len(indices)}),flush=True)
        # events.jsonl is an execution log, not editable after freeze. Its final event
        # is stored separately below to avoid mutating the already hashed stream.
        return frozen
    except Exception as exc:
        write_json(root/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),
            'completed_replicates':indices,'elapsed_seconds':time.monotonic()-started})
        raise


def evaluate_experiment(output):
    root=Path(output).resolve();frozen=verify_seal(root)
    out=root/'final_evaluation'
    if out.exists():raise ValueError('Final evaluation already exists; never tune or overwrite it')
    out.mkdir();index=json.loads((root/'INDEX.json').read_text());all_rows=[]
    torch.set_num_threads(1)
    for rel in index['replicates']:
        rep=json.loads((root/rel).read_text());split_path=root/rep['split_path'];meta=json.loads((split_path/'metadata.json').read_text())
        test=np.load(split_path/'test.npz',allow_pickle=False)
        for row in rep['rows']:
            path=root/row['path']
            p=(load_student(path).predict_proba(test['x']) if row['student']=='mlp' else
               logistic_predict(json.loads((path/'model.json').read_text()),test['x']))
            tag=canonical_hash({'path':row['path']})[:24]
            np.savez_compressed(out/f'{tag}.npz',ids=test['ids'],labels=test['y'],probabilities=p)
            all_rows.append({**row,'dataset':meta['dataset'],'split_seed':meta['split_seed'],
                'generation_seed':rep['generation_seed'],'test':classification_metrics(test['y'],p),
                'prediction_path':str((out/f'{tag}.npz').relative_to(root)),
                'selected_by_development':row['method']==rep['selection']['method'] and row['regime']==rep['selection']['regime']})
    write_json(out/'scores.json',{'frozen_manifest_sha256':frozen['manifest_sha256'],'rows':all_rows,
        'test_used_for_selection':False,'no_further_tuning_permitted':True})
    files=[str(p.relative_to(out)) for p in out.iterdir() if p.is_file()]
    seal(out,files)
    return {'scored_conditions':len(all_rows),'frozen_manifest_sha256':frozen['manifest_sha256']}
