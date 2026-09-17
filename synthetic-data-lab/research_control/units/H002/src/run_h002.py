"""Full data dependence probe using immutable H001 and B002 evidence."""
import argparse,sys,json,os,shutil,time
from pathlib import Path
import numpy as np
from rank_synthesis import generate


def main(args):
    support=args.support.resolve()
    sys.path[:0]=[str(support/'units/B002/src'),str(support/'units/H001/src'),str(support/'units/H001C/src')]
    from run import write,verify_input,fit_linear,sha
    from run_h001 import fit_student
    from calibration import benchmark
    out=args.output.resolve()
    if out.exists():raise ValueError('Existing output refused')
    out.mkdir(parents=True)
    write(out/'START.json',{'unit':'H002','seed':args.seed,'source_commit':os.environ.get('GITHUB_SHA'),'run_id':os.environ.get('GITHUB_RUN_ID'),'final_test_scored':False})
    try:
        a_hash=verify_input(args.input);b_hash=verify_input(args.baseline)
        t=np.load(args.input/'run/data/train.npy',allow_pickle=False);d=np.load(args.input/'run/data/dev.npy',allow_pickle=False)
        if t.shape!=(348563,13) or d.shape!=(116314,13):raise ValueError('Full source partition required')
        with np.load(args.baseline/'run/generator.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
        import hashlib
        if str(model['input_sha256'])!=hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest():raise ValueError('Fitted baseline input mismatch')
        write(out/'INPUT.json',{'h001_manifest_sha256':a_hash,'b002_manifest_sha256':b_hash,'train_file_sha256':sha(args.input/'run/data/train.npy'),'dev_file_sha256':sha(args.input/'run/data/dev.npy'),'training_rows':len(t),'development_rows':len(d)})
        results={}
        for method in ('gaussian_rank','beta_rank','beta_wide_rank','independent_rank'):
            folder=out/method;folder.mkdir();start=time.monotonic()
            a,lineage=generate(t,model,args.seed,method)
            np.save(folder/'synthetic.npy',a,allow_pickle=False);np.savez_compressed(folder/'generation.npz',**lineage)
            write(folder/'GENERATION.json',{'seconds':time.monotonic()-start,'rows':len(a),'method':method,'seed':args.seed,'kind':'new combinations of empirical values','rejected_rows':0})
            q=benchmark(a,d,t);write(folder/'QUALITY.json',q)
            tree=fit_student(a,d,folder/'tree',args.seed)
            write(out/'PROGRESS.json',{'method':method,'stage':'tree_complete'})
            linear=fit_linear(a,d,folder/'linear')
            results[method]={'tree':tree,'linear':linear,'seconds':time.monotonic()-start,'quality':{k:v for k,v in q.items() if k!='class_conditional_tail_events'}}
            write(folder/'RESULT.json',results[method]);write(out/'PROGRESS.json',{'completed':list(results)})
            print(json.dumps({'method':method,'tree':tree['balanced_accuracy'],'linear':linear['balanced_accuracy'],'copies':q['copy_count']}),flush=True)
        write(out/'SUMMARY.json',results)
    except Exception as exc:
        write(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--seed',type=int,required=True);main(p.parse_args())
