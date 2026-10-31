"""Full population training only H004A calibration and fresh generation audit.

Calibration and audit are separate commands. The audit never changes selection.
No development data reader or model fitting function is present in this module.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from mechanism import (QUANTILES, atomic_json, candidate_widths, choose_width,
                       sample_context, beta_variance_ratio)

CALIBRATION_SEEDS = (41017, 41029)
AUDIT_SEEDS = (43017, 43029, 43041)
METHODS = ('unit_beta', 'heuristic', 'empirical_median', 'empirical_quantile')
EXPECTED_TRAIN_SHA = 'c39bf8a33ec369f8472622623251d1c4aa9082ce4a582397c311599f8f653418'
SOURCE_PINS = {
 'H002/src/rank_synthesis.py':'c5164c62e9ea1e54321d0b885b88a1432f8514a6f5a0735c767f35c1a86d26c9',
 'H003/src/neighborhood.py':'b377979d25670b8e99e225542ab4f8a3d96f4626dcb448b7da9257122d5786fb',
 'B002/src/copula.py':'c26c32b2c0c1ec479acb586749752f41bf7a0b24d2d8fee1729ce54894e9800f',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.h004a_', suffix='.npz', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def compact(table):
    require(table.dtype.kind in 'iu' and np.all(table >= np.iinfo(np.int32).min)
            and np.all(table <= np.iinfo(np.int32).max), 'Lossless int32 storage required')
    return table.astype(np.int32)


def load_inputs(root):
    root = Path(root).resolve()
    paths = {
        'training':root/'h001-seed17/run/data/train.npy',
        'contexts':root/'b002-seed17/run/generator.npz',
        'geometry':root/'h003-seed17/run/geometry.npz',
    }
    for folder in ('h001-seed17', 'b002-seed17', 'h003-seed17'):
        manifest = json.loads((root/folder/'ARTIFACT_SHA256.json').read_text())
        for name, expected in manifest.items():
            p = root/folder/name
            require(p.resolve().is_relative_to(root/folder) and not p.is_symlink(), 'Unsafe input')
            require(digest(p) == expected, f'Input hash mismatch: {folder}/{name}')
    require(digest(paths['training']) == EXPECTED_TRAIN_SHA, 'Wrong training file identity')
    t = np.load(paths['training'], allow_pickle=False)
    require(t.shape == (348563, 13) and t.dtype == np.dtype('int64'), 'Wrong population/schema')
    with np.load(paths['contexts'], allow_pickle=False) as z:
        m = {k:z[k] for k in z.files}
    with np.load(paths['geometry'], allow_pickle=False) as z:
        geo = {k:z[k] for k in ('scale','loo_distance','bandwidth','pilot_median','target_median')}
    require(len(m['counts']) == 232 and m['contexts'].shape == (232,3), 'Wrong context family')
    require(np.array_equal(m['offsets'], np.r_[0,np.cumsum(m['counts'])]), 'Offsets corrupted')
    require(np.array_equal(np.sort(m['group_indices']), np.arange(len(t))), 'Membership corrupted')
    require(str(m['input_sha256'].item()) == hashlib.sha256(t.tobytes()).hexdigest(), 'Context input binding')
    require(np.array_equal(geo['scale'], np.maximum(t[:,:10].std(0),1e-8)), 'Scale mismatch')
    for g,n in enumerate(m['counts']):
        sl = slice(m['offsets'][g], m['offsets'][g+1]); ids = m['group_indices'][sl]
        require(np.all(t[ids,10:] == m['contexts'][g]), 'Context identities mismatch')
        require(np.array_equal(np.sort(t[ids,:10],axis=0),m['sorted'][sl]), 'Context marginal mismatch')
    units = Path(__file__).resolve().parents[2]
    for name,h in SOURCE_PINS.items():
        require(digest(units/name) == h, f'Frozen source changed: {name}')
    return t,m,geo,{name:{'path':str(p),'sha256':digest(p)} for name,p in paths.items()}


def environment():
    return {'python':sys.version,'platform':platform.platform(),
            'libraries':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn')},
            'thread_controls':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'argv':sys.argv,'development_values_read':False,'reserved_values_read':False,
            'protocol_commit':'946c0fd5d604f255cf3fd60104707946149510fa',
            'new_source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
            'dependency_sha256':SOURCE_PINS}


def nearest_with_tree(tree, x, scale):
    d = tree.query(x/scale,k=1,workers=1)[0]
    require(np.isfinite(d).all() and np.all(d>=0), 'Invalid nearest distance')
    return d


def context_calibration(g,t,m,geo):
    n = int(m['counts'][g]); sl = slice(m['offsets'][g],m['offsets'][g+1]); ids=m['group_indices'][sl]
    x=t[ids,:10]; scale=geo['scale']; tree=cKDTree(x/scale)
    target=geo['loo_distance'][ids]
    fresh=np.zeros(1) if n==1 else tree.query(x/scale,k=2,workers=1)[0][:,1]
    require(np.allclose(fresh,target,rtol=0,atol=1e-12), 'Frozen leave one out distances not reproduced')
    tq=np.quantile(target,QUANTILES)
    require(abs(tq[2]-geo['target_median'][g]) < 1e-12, 'Target median identity mismatch')
    h=float(geo['bandwidth'][g]); widths=candidate_widths(n,h)
    arrays={'ids':ids,'widths':widths,'target_distances':target,'target_quantiles':tq,
            'seeds':np.array(CALIBRATION_SEEDS),'context':m['contexts'][g],
            'heuristic':np.array(h),'pilot_median':np.array(geo['pilot_median'][g])}
    tables=[]; distances=[]; inverses=[]; anchor_quantiles=[]; inverse_anchor_quantiles=[]
    for seed in CALIBRATION_SEEDS:
        tt=[]; dd=[]; ii=[]; aa=[]; ia=[]
        for f in widths:
            draw=sample_context(x,ids,float(f),seed,g)
            a,b=draw['assigned'],draw['inverse']
            require(np.array_equal(np.sort(a,axis=0),np.sort(x,axis=0)), 'Marginal contract failed')
            tt.append(compact(a)); dd.append(nearest_with_tree(tree,a,scale)); ii.append(nearest_with_tree(tree,b,scale))
            aa.append(np.quantile(np.linalg.norm((a-x)/scale,axis=1),QUANTILES))
            ia.append(np.quantile(np.linalg.norm((b-x)/scale,axis=1),QUANTILES))
        tables.append(tt); distances.append(dd); inverses.append(ii)
        anchor_quantiles.append(aa); inverse_anchor_quantiles.append(ia)
    arrays.update(tables=np.array(tables), distances=np.array(distances), inverse_distances=np.array(inverses),
                  anchor_quantiles=np.array(anchor_quantiles), inverse_anchor_quantiles=np.array(inverse_anchor_quantiles))
    # np.quantile returns (quantile,seed,width); transpose to the contract shape.
    q=np.quantile(arrays['distances'],QUANTILES,axis=2).transpose(1,2,0)
    iq=np.quantile(arrays['inverse_distances'],QUANTILES,axis=2).transpose(1,2,0)
    arrays.update(quantiles=q,inverse_quantiles=iq)
    record={'context_index':g,'context':m['contexts'][g].tolist(),'n':n,'heuristic':h,
            'widths':widths.tolist(),'target_quantiles':tq.tolist(),
            'empirical_median':choose_width(widths,q,tq,'median'),
            'empirical_quantile':choose_width(widths,q,tq,'quantile'),
            'primary_population':bool(n>=100 and tq[2]>0),
            'zero_target':bool(tq[2]==0),'original_grid_count':6,'unique_grid_count':len(widths)}
    return arrays,record


def calibrate(args):
    out=args.output.resolve()
    require(not out.exists(), 'Refuse existing calibration output')
    require(not out.is_relative_to(args.inputs.resolve()), 'Keep output outside immutable inputs')
    t,m,geo,pins=load_inputs(args.inputs)
    out.mkdir(parents=True)
    atomic_json(out/'START.json',{'environment':environment(),'inputs':pins,'training_rows':len(t),'contexts':len(m['counts'])})
    records=[]; start=time.monotonic()
    for g in range(len(m['counts'])):
        require(time.monotonic()-start < 1800, 'Calibration wall time ceiling reached')
        arrays,record=context_calibration(g,t,m,geo)
        file=out/f'context_{g:03d}.npz';save_npz(file,**arrays);record['sha256']=digest(file)
        records.append(record)
        print(json.dumps({'stage':'calibration','context':g,'rows':record['n'],
                          'settings':record['unique_grid_count'],'seconds':time.monotonic()-start}),flush=True)
    atomic_json(out/'SELECTION.json',{'calibration_seeds':CALIBRATION_SEEDS,'audit_seeds':AUDIT_SEEDS,
               'quantile_levels':QUANTILES.tolist(),'records':records,'inputs':pins,
               'status':'frozen_before_audit','training_rows':len(t),'elapsed_seconds':time.monotonic()-start})
    atomic_json(out/'COMPLETE.json',{'selection_sha256':digest(out/'SELECTION.json'),
                                  'contexts':len(records),'elapsed_seconds':time.monotonic()-start})


def ks_distance(a,b):
    a=np.sort(a);b=np.sort(b);both=np.r_[a,b]
    return float(np.max(np.abs(np.searchsorted(a,both,side='right')/len(a)-np.searchsorted(b,both,side='right')/len(b))))


def spearman_error(x,y):
    varying=np.std(x,axis=0)>0
    if varying.sum()<2:return 0.0
    def corr(a):
        z=rankdata(a[:,varying],axis=0);z=(z-z.mean(0))/z.std(0)
        return z.T@z/len(z)
    a,b=corr(x),corr(y);mask=np.triu(np.ones(a.shape,dtype=bool),1)
    return float(np.abs(a-b)[mask].mean())


def audit(args):
    out=args.output.resolve();cal=args.calibration.resolve()
    require(not out.exists(),'Refuse existing audit output')
    require(not out.is_relative_to(args.inputs.resolve()) and not out.is_relative_to(cal),
            'Keep audit output outside immutable inputs and calibration')
    t,m,geo,pins=load_inputs(args.inputs)
    selection=json.loads((cal/'SELECTION.json').read_text());sealed=json.loads((cal/'COMPLETE.json').read_text())
    selection_hash=digest(cal/'SELECTION.json')
    require(sealed['selection_sha256']==selection_hash,'Selection was modified')
    require(selection['calibration_seeds']==list(CALIBRATION_SEEDS) and selection['audit_seeds']==list(AUDIT_SEEDS),'Seed split changed')
    require(selection['inputs']==pins,'Input binding mismatch')
    require(len(selection['records'])==232,'Incomplete selection')
    for row in selection['records']:
        require(digest(cal/f"context_{row['context_index']:03d}.npz")==row['sha256'],'Calibration evidence changed')
    out.mkdir(parents=True);atomic_json(out/'START.json',{'environment':environment(),'inputs':pins,
                     'selection_sha256':selection_hash,'stage':'audit','training_rows':len(t)})
    reports=[];start=time.monotonic()
    for seed in AUDIT_SEEDS:
        for method in METHODS:
            tables=[];ds=[];rows=[]
            for g,sel in enumerate(selection['records']):
                require(time.monotonic()-start < 1800,'Audit wall time ceiling reached')
                n=int(m['counts'][g]);sl=slice(m['offsets'][g],m['offsets'][g+1]);ids=m['group_indices'][sl]
                x=t[ids,:10];scale=geo['scale'];target=geo['loo_distance'][ids];tq=np.quantile(target,QUANTILES)
                width=1. if method=='unit_beta' else float(sel[method])
                draw=sample_context(x,ids,width,seed,g);numeric=draw['assigned']
                require(np.array_equal(np.sort(numeric,axis=0),m['sorted'][sl]),'Audit marginal violation')
                table=np.column_stack((numeric,np.broadcast_to(m['contexts'][g],(n,3))))
                tree=cKDTree(x/scale);d=nearest_with_tree(tree,numeric,scale);q=np.quantile(d,QUANTILES)
                row={'context_index':g,'class':int(m['contexts'][g,2]),'n':n,'width':width,
                     'primary_population':bool(n>=100 and tq[2]>0),'target_quantiles':tq.tolist(),'quantiles':q.tolist(),
                     'median_error':float(abs(q[2]-tq[2])/tq[2]) if tq[2]>0 else None,
                     'quantile_error':float(np.mean(abs(q-tq))/tq[2]) if tq[2]>0 else None,
                     'median_ratio':float(q[2]/tq[2]) if tq[2]>0 else None,
                     'ks_distance':ks_distance(d,target),'below_q05_count':int((d<tq[0]).sum()),
                     'below_median_count':int((d<tq[2]).sum()),'zero_distance_count':int((d==0).sum()),
                     'unique_rows':len(np.unique(numeric,axis=0)),'spearman_error':spearman_error(x,numeric),
                     'bandwidth_on_upper_grid':bool(width==max(sel['widths']))}
                rows.append(row);tables.append(compact(table));ds.append(d)
            table=np.concatenate(tables);distances=np.concatenate(ds)
            dest=out/f'{method}_{seed}.npz';save_npz(dest,table=table,distances=distances,counts=m['counts'],contexts=m['contexts'])
            primary=[r for r in rows if r['primary_population']];den=sum(r['n'] for r in primary)
            def weighted(key,subset=primary):return sum(r['n']*r[key] for r in subset)/sum(r['n'] for r in subset)
            summary={'method':method,'seed':seed,'training_rows':len(t),'primary_rows':den,'primary_contexts':len(primary),
                     'primary_coverage':den/len(t),'weighted_median_error':weighted('median_error'),
                     'weighted_quantile_error':weighted('quantile_error'),'weighted_median_ratio':weighted('median_ratio'),
                     'weighted_ks':weighted('ks_distance'),'weighted_spearman_error':weighted('spearman_error'),
                     'all_zero_distance_count':sum(r['zero_distance_count'] for r in rows),
                     'all_zero_distance_fraction':sum(r['zero_distance_count'] for r in rows)/len(t),
                     'primary_below_q05_fraction':sum(r['below_q05_count'] for r in primary)/den,
                     'primary_below_median_fraction':sum(r['below_median_count'] for r in primary)/den,
                     'all_unique_rows':sum(r['unique_rows'] for r in rows),
                     'artifact':dest.name,'artifact_sha256':digest(dest),'contexts':rows}
            atomic_json(out/f'{method}_{seed}.json',summary);reports.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='contexts'}),flush=True)
    screen={}
    for method in ('empirical_median','empirical_quantile'):
        cases=[]
        for seed in AUDIT_SEEDS:
            base=next(r for r in reports if r['method']=='heuristic' and r['seed']==seed)
            cand=next(r for r in reports if r['method']==method and r['seed']==seed)
            reduction=1-cand['weighted_median_error']/base['weighted_median_error']
            passed=reduction>=.30 and cand['all_zero_distance_fraction']<=base['all_zero_distance_fraction']+.001
            cases.append({'seed':seed,'median_error_reduction':reduction,'screen_pass':passed})
        screen[method]={'seeds':cases,'all_seed_screen_pass':all(c['screen_pass'] for c in cases)}
    require(digest(cal/'SELECTION.json')==selection_hash,'Selection changed during audit')
    atomic_json(out/'SUMMARY.json',{'results':[{k:v for k,v in r.items() if k!='contexts'} for r in reports],
                'mechanism_screen':screen,'selection_sha256':selection_hash,'selection_unchanged':True,
                'elapsed_seconds':time.monotonic()-start,'scientific_novelty_established':False,
                'development_scored':False,'reserved_scored':False})
    atomic_json(out/'COMPLETE.json',{'status':'completed_awaiting_independent_verification',
                                   'summary_sha256':digest(out/'SUMMARY.json')})


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='stage',required=True)
    for name in ('calibrate','audit'):
        p=sub.add_parser(name);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if name=='audit':p.add_argument('--calibration',type=Path,required=True)
    args=parser.parse_args()
    if sys.flags.optimize:raise RuntimeError('Optimized Python is not an accepted validation runtime')
    (calibrate if args.stage=='calibrate' else audit)(args)


if __name__=='__main__':main()
