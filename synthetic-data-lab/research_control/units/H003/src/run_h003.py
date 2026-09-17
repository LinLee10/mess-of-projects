import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np


def main(a):
    support=a.support.resolve()
    sys.path[:0]=[str(support/'units/H002/src'),str(support/'units/B002/src'),str(support/'units/H001/src'),str(support/'units/H001C/src')]
    from run import verify_input,write,fit_linear
    from neighborhood import fit_geometry,draw
    from proximity import nearest,summarize
    from run_h001 import fit_student
    from calibration import benchmark
    out=a.output
    if out.exists():raise ValueError('Existing output')
    out.mkdir(parents=True)
    write(out/'START.json',{'unit':'H003','seed':a.seed,'source_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),'final_test_scored':False})
    try:
        ih=verify_input(a.input);bh=verify_input(a.baseline)
        t=np.load(a.input/'run/data/train.npy',allow_pickle=False);d=np.load(a.input/'run/data/dev.npy',allow_pickle=False)
        if t.shape!=(348563,13) or d.shape!=(116314,13):raise ValueError('Unexpected full input sizes')
        with np.load(a.baseline/'run/generator.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
        th=hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()
        if th!=str(m['input_sha256']):raise ValueError('Training identity mismatch')
        write(out/'INPUT.json',{'h001_manifest_sha256':ih,'b002_manifest_sha256':bh,'training_bytes_sha256':th,'training_rows':len(t),'development_rows':len(d)})
        start=time.monotonic();geometry=fit_geometry(t,m);np.savez_compressed(out/'geometry.npz',**geometry)
        write(out/'FIT.json',{'seconds':time.monotonic()-start,'data_access':'Training only','bandwidth_formula':'clip((median training leave one out distance / median standard beta pilot nearest training distance)^2,1,context_count)','pilot_seed_base':24017,'contexts':len(m['counts'])})
        rd,ri=nearest(t,d,geometry['scale']);np.savez_compressed(out/'real_development_distances.npz',distance=rd,nearest_training_index=ri)
        results={}
        for method in ('neighbor_raw','neighbor_rank','adaptive_beta'):
            folder=out/method;folder.mkdir();start=time.monotonic()
            table,lineage=draw(t,m,geometry,a.seed,method);np.save(folder/'synthetic.npy',table,allow_pickle=False);np.savez_compressed(folder/'generation.npz',**lineage)
            q=benchmark(table,d,t);write(folder/'QUALITY.json',q)
            tree=fit_student(table,d,folder/'tree',a.seed);write(out/'PROGRESS.json',{'stage':'tree_completed','method':method})
            linear=fit_linear(table,d,folder/'linear')
            sd,si=nearest(t,table,geometry['scale']);np.savez_compressed(folder/'proximity.npz',distance=sd,nearest_training_index=si);pr=summarize(sd,rd);write(folder/'PROXIMITY.json',pr)
            results[method]={'tree':tree,'linear':linear,'quality':{k:v for k,v in q.items() if k!='class_conditional_tail_events'},'proximity':pr,'seconds':time.monotonic()-start}
            write(folder/'RESULT.json',results[method]);write(out/'PROGRESS.json',{'completed':list(results)})
            print(json.dumps({'method':method,'tree_balanced_accuracy':tree['balanced_accuracy'],'linear_balanced_accuracy':linear['balanced_accuracy'],'near_reference_fraction':pr['strictly_below_real_q05_fraction']}),flush=True)
        write(out/'SUMMARY.json',results)
    except Exception as exc:
        write(out/'FAILED.json',{'type':type(exc).__name__,'message':str(exc)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--seed',type=int,required=True);main(p.parse_args())
