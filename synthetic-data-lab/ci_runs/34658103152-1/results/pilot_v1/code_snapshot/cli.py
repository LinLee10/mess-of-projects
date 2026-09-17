"""Explicit training, evaluation, analysis and exported model inference."""
import argparse,json
from .experiment import train_experiment,evaluate_experiment

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--output',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--run',required=True);a.add_argument('--output',required=True)
    s=sub.add_parser('sample');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True)
    s.add_argument('--count',type=int,default=1000);s.add_argument('--seed',type=int,default=17)
    q=sub.add_parser('predict');q.add_argument('--checkpoint',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    b=sub.add_parser('screen-wine');b.add_argument('--input',required=True);b.add_argument('--schema',required=True);b.add_argument('--output',required=True)
    v=sub.add_parser('verify');v.add_argument('--run',required=True)
    args=p.parse_args()
    if args.command=='train':result=train_experiment(args.config,args.output)
    elif args.command=='evaluate':result=evaluate_experiment(args.output)
    elif args.command=='analyze':
        from .analysis import analyze
        result=analyze(args.run,args.output)
    elif args.command=='sample':
        from .inference import sample_csv
        result=sample_csv(args.checkpoint,args.output,args.count,args.seed)
    elif args.command=='screen-wine':
        from .quality import screen_wine_csv
        result=screen_wine_csv(args.input,args.schema,args.output)
    elif args.command=='predict':
        from .inference import predict_csv
        result=predict_csv(args.checkpoint,args.input,args.output)
    else:
        from pathlib import Path
        from .common import verify_seal
        result={'candidate':verify_seal(args.run)['manifest_sha256'],
                'evaluation':verify_seal(Path(args.run)/'final_evaluation')['manifest_sha256']}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
