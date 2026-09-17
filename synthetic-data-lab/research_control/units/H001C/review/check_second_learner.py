"""Separate linear inference and metric checker. No worker imports."""
import argparse,json
from pathlib import Path
import numpy as np


def check(root,input_path):
    dev=np.load(Path(input_path)/'run/data/dev.npy',allow_pickle=False)
    checked=[]
    for name in ['real','raw','pooled_legacy','pooled_keyed','context_keyed']:
        p=Path(root)/name;m=json.loads((p/'model.json').read_text());s=json.loads((p/'RESULT.json').read_text())
        if not m['converged']:raise ValueError('Incomplete optimization')
        with np.load(p/'predictions.npz',allow_pickle=False) as z:y,recorded=z['labels'],z['probabilities']
        if not np.array_equal(y,dev[:,12]) or not np.isfinite(recorded).all():raise ValueError('Bad saved outcomes')
        x=np.zeros((len(dev),54));x[:,:10]=(dev[:,:10]-m['mean'])/m['scale']
        x[np.arange(len(dev)),10+dev[:,10]]=1;x[np.arange(len(dev)),14+dev[:,11]]=1
        scores=x@np.array(m['coef']).T+np.array(m['intercept']);scores-=scores.max(1,keepdims=True)
        predicted=np.exp(scores);predicted/=predicted.sum(1,keepdims=True)
        error=float(np.max(np.abs(predicted-recorded)))
        if not np.allclose(predicted,recorded,rtol=1e-9,atol=1e-10) or not np.array_equal(predicted.argmax(1),recorded.argmax(1)):raise ValueError('Linear inference mismatch')
        cm=np.array([np.bincount(recorded[y==c].argmax(1),minlength=7) for c in range(7)])
        ba=sum(cm[i,i]/cm[i].sum() for i in range(7))/7
        if cm.tolist()!=s['utility']['confusion_matrix'] or abs(ba-s['utility']['balanced_accuracy'])>1e-12:raise ValueError('Independent metric mismatch')
        checked.append({'variant':name,'maximum_probability_error':error,'count':len(y),'balanced_accuracy':ba})
    return {'status':'passed','production_imports':False,'independence':'Same assistant; independent code and process','checked':checked}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Refuse to overwrite review')
    a.output.write_text(json.dumps(check(a.run,a.input),indent=2)+'\n')
