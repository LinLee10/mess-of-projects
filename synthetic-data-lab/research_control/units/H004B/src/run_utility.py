"""Frozen development utility challenge; no reserved test reader or tuning loop."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,platform,importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'H004A/src'),str(ROOT/'H002/src'),str(ROOT/'B002/src'),str(ROOT/'H001/src'),str(ROOT/'H001C/src')]
from mechanism import atomic_json, QUANTILES
from run_response import load_inputs,save_npz,digest,ks_distance,spearman_error
from rank_synthesis import ordinal,assign_columns
from run_h001 import fit_student
from run import fit_linear
from scipy.spatial import cKDTree
from resampling import iid_context,dependence_error
SEEDS=(43017,43029,43041)
METHODS=('unit_beta','heuristic','empirical_median','empirical_quantile','iid_beta_inverse','iid_beta_rank')
PINS={'H003R/review/audit_recovery.py':'d1ffc4c1c448cfb30a07c919a2942257dd9c2ca7de3e7a610d5524a6a6220569','H001/src/run_h001.py':'82c9375cbc6ab48b3098e125dfa3d580b7079b745f8761fd98ad14910d4babe1','H001C/src/second_learner.py':'c77d16dfefe7f1a3091aa93b593a3f29d0f229e081f2d5f487e9637473770bb1','B002/src/run.py':'25fa6f9f4dd2d33a44eb6e9694d8534b8317c4b1b8376e42681df7f01442b86a'}
def require(ok,msg):
    if not ok:raise ValueError(msg)


def generate_references(t,m,geo,seed,out):
    inv=[];rank=[];latent=[];anchors=[];idslist=[]
    for g,n in enumerate(m['counts']):
        ids=m['group_indices'][m['offsets'][g]:m['offsets'][g+1]];x=t[ids,:10]
        z=iid_context(x,ids,int(seed),int(g));ctx=np.broadcast_to(m['contexts'][g],(len(ids),3))
        inv.append(np.column_stack((z['inverse'],ctx)));rank.append(np.column_stack((z['assigned'],ctx)))
        latent.append(z['latent']);anchors.append(z['anchor_ids']);idslist.append(ids)
    save_npz(out/f'paper_lineage_{seed}.npz',latent=np.concatenate(latent),anchor_ids=np.concatenate(anchors),original_ids=np.concatenate(idslist),contexts=m['contexts'],counts=m['counts'])
    result={'iid_beta_inverse':np.concatenate(inv),'iid_beta_rank':np.concatenate(rank)}
    for method,a in result.items():
        # Reuse matched original training geometry, never development values.
        rows=[];distances=[]
        for g,n in enumerate(m['counts']):
            sl=slice(m['offsets'][g],m['offsets'][g+1]);ids=m['group_indices'][sl];x=t[ids,:10];y=a[sl,:10];target=geo['loo_distance'][ids]
            require(np.array_equal(a[sl,10:],t[ids,10:]),'Changed context frequencies')
            d=cKDTree(x/geo['scale']).query(y/geo['scale'],k=1,workers=1)[0];q=np.quantile(d,QUANTILES);tq=np.quantile(target,QUANTILES)
            row={'g':g,'n':int(n),'class':int(m['contexts'][g,2]),'primary':bool(n>=100 and tq[2]>0),'median_error':float(abs(q[2]-tq[2])/tq[2]) if tq[2]>0 else None,'quantile_error':float(np.mean(abs(q-tq))/tq[2]) if tq[2]>0 else None,'ks_distance':ks_distance(d,target),'spearman_error':dependence_error(x,y)['mean_absolute_error'],'collapsed_synthetic_columns':dependence_error(x,y)['collapsed_synthetic_columns'],'zero_count':int((d==0).sum()),'unique':len(np.unique(y,axis=0)),'exact_marginals':bool(np.array_equal(np.sort(x,axis=0),np.sort(y,axis=0))),'marginal_ks':float(np.mean([ks_distance(x[:,j],y[:,j]) for j in range(10)]))}
            if method=='iid_beta_rank':require(row['exact_marginals'],'Rank reference marginal failure')
            rows.append(row);distances.append(d)
        save_npz(out/f'{method}_{seed}.npz',table=a.astype(np.int32),distances=np.concatenate(distances),counts=m['counts'],contexts=m['contexts'])
        primary=[r for r in rows if r['primary']];den=sum(r['n'] for r in primary)
        meta={'method':method,'seed':seed,'training_rows':len(t),'primary_rows':den,'zero_distance_count':sum(r['zero_count'] for r in rows),'unique_rows':sum(r['unique'] for r in rows),'weighted_marginal_ks':sum(r['n']*r['marginal_ks'] for r in rows)/len(t),'weighted_median_error':sum(r['n']*r['median_error'] for r in primary)/den,'weighted_quantile_error':sum(r['n']*r['quantile_error'] for r in primary)/den,'weighted_spearman_error':sum(r['n']*r['spearman_error'] for r in primary)/den,'contexts':rows}
        atomic_json(out/f'{method}_{seed}.json',meta)
    return result

def check_saved(folder,dev):
    # Reuse the frozen separate arithmetic and inference checker, not production scores.
    import importlib.util
    path=ROOT/'H003R/review/audit_recovery.py'
    spec=importlib.util.spec_from_file_location('historical_readonly_checker',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return {learner:mod.check_model(folder/learner,dev) for learner in ('tree','linear')}

def main(args):
    out=args.output.resolve();require(not out.exists(),'Refuse existing output')
    for p in (args.inputs,args.audit,args.verification):require(not out.is_relative_to(p.resolve()),'Output must not overwrite input')
    report=json.loads(args.verification.read_text());require(report.get('status')=='verified' and not report.get('fixture',True),'Full independent H004A verification required')
    counts=report['counts'];require(counts['training_rows']==348563 and counts['contexts']==232 and counts['audit_files']==12 and counts['audit_rows']==348563*12,'Incomplete H004A verification')
    # The verifier binds every original H004A generated file. Recheck those bytes before reuse.
    for p,h in report['artifact_sha256'].items():require(digest(p)==h,'Previously verified evidence changed: '+p)
    for name,h in PINS.items():require(digest(ROOT/name)==h,'Frozen student source changed')
    t,m,geo,pins=load_inputs(args.inputs)
    devfile=args.inputs/'h001-seed17/run/data/dev.npy';dev=np.load(devfile,allow_pickle=False)
    require(dev.shape==(116314,13) and dev.dtype==np.dtype('int64'),'Wrong full development table')
    out.mkdir(parents=True);atomic_json(out/'START.json',{'unit':'H004B','protocol_commit':'5902ab9ba70d99810038895f505056e81473da3f','h004a_verification_sha256':digest(args.verification),'inputs':pins,'development_sha256':digest(devfile),'argv':sys.argv,'source_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},'convention_commit':'370fe3ea02c1a3e27591ab187941b7d7ce41cd6e','dependency_sha256':PINS,'python':sys.version,'libraries':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','lightgbm']},'reserved_data_access':False,'generation_seeds':SEEDS,'fit_count_planned':38})
    start=time.monotonic();order=np.random.default_rng(88017).permutation(len(t));np.save(out/'student_row_order.npy',order,allow_pickle=False)
    results=[]
    def fit(method,seed,table):
        require(time.monotonic()-start<1800,'Full utility measurement ceiling reached')
        require(table.shape==t.shape and table.dtype.kind in 'iu','Wrong full training table')
        dest=out/f'{method}_{seed}';dest.mkdir();a=table[order].astype(np.int64)
        tree=fit_student(a,dev,dest/'tree',seed);linear=fit_linear(a,dev,dest/'linear')
        checked=check_saved(dest,dev)
        item={'method':method,'seed':seed,'tree':tree,'linear':linear,'checked':checked,'training_rows':len(a),'development_rows':len(dev),'student_table_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        atomic_json(dest/'RESULT.json',item);results.append(item);print(json.dumps({'method':method,'seed':seed,'tree_balanced_accuracy':tree['balanced_accuracy'],'linear_balanced_accuracy':linear['balanced_accuracy'],'seconds_total':time.monotonic()-start}),flush=True)
    fit('real',0,t)
    for seed in SEEDS:
        references=generate_references(t,m,geo,seed,out)
        for method in METHODS:
            if method in references:a=references[method]
            else:
                with np.load(args.audit/f'{method}_{seed}.npz',allow_pickle=False) as z:a=z['table']
            fit(method,seed,a)
    screen={}
    for method in ('empirical_median','empirical_quantile'):
        c=[r for r in results if r['method']==method];h=[r for r in results if r['method']=='heuristic']
        diffs={learner:float(np.mean([x[learner]['balanced_accuracy']-y[learner]['balanced_accuracy'] for x,y in zip(c,h)])) for learner in ('tree','linear')}
        recalls={learner:np.mean([np.array(x[learner]['class_recall'])-y[learner]['class_recall'] for x,y in zip(c,h)],axis=0).tolist() for learner in ('tree','linear')}
        geometry=report['mechanism_screen'][method]
        if isinstance(geometry,dict):geometry=geometry['all_seed_screen_pass']
        passed=bool(geometry and diffs['tree']>=-.01 and diffs['linear']>=-.005 and min(recalls['tree'])>=-.03 and min(recalls['linear'])>=-.03)
        screen[method]={'mean_balanced_accuracy_change':diffs,'mean_class_recall_change':recalls,'geometry_screen':geometry,'exploratory_engineering_screen_pass':passed}
    atomic_json(out/'SUMMARY.json',{'results':results,'engineering_screen':screen,'elapsed_seconds':time.monotonic()-start,'fresh_student_fits':len(results)*2,'reserved_scored':False,'scientific_novelty_established':False})
    manifest={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    atomic_json(out/'MANIFEST.json',manifest);atomic_json(out/'COMPLETE.json',{'status':'completed','manifest_sha256':digest(out/'MANIFEST.json'),'fresh_student_fits':len(results)*2})
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','audit','verification','output'):p.add_argument('--'+name,type=Path,required=True)
    if sys.flags.optimize:raise RuntimeError('Optimized validation is forbidden')
    main(p.parse_args())
