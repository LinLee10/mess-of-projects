import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from second_learner import encode,summarize
from check_second_learner import check

class SecondLearnerControls(unittest.TestCase):
    def test_categories_expand_to_fixed_columns(self):
        a=np.zeros((2,13),dtype=np.int64);a[1,10]=3;a[1,11]=39
        x=encode(a,np.zeros(10),np.ones(10))
        self.assertEqual(x.shape,(2,54));self.assertEqual(x[1,13],1);self.assertEqual(x[1,53],1)
        self.assertTrue(np.all(x[:,10:].sum(1)==2))
    def test_target_never_used_as_feature(self):
        a=np.zeros((2,13),dtype=np.int64);b=a.copy();b[:,12]=6
        self.assertTrue(np.array_equal(encode(a,np.zeros(10),np.ones(10)),encode(b,np.zeros(10),np.ones(10))))
    def test_unknown_category_refused(self):
        a=np.zeros((2,13),dtype=np.int64);a[0,10]=4
        with self.assertRaises(ValueError):encode(a,np.zeros(10),np.ones(10))
    def test_checker_and_tamper_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'input/run/data').mkdir(parents=True)
            a=np.zeros((7,13),dtype=np.int64);a[:,12]=np.arange(7)
            np.save(base/'input/run/data/dev.npy',a);coef=np.zeros((7,54));inter=np.arange(7,dtype=float)
            p=np.exp(inter-inter.max());p=np.tile(p/p.sum(),(7,1))
            for name in ['real','raw','pooled_legacy','pooled_keyed','context_keyed']:
                r=base/'out'/name;r.mkdir(parents=True)
                (r/'model.json').write_text(json.dumps({'mean':[0]*10,'scale':[1]*10,'coef':coef.tolist(),'intercept':inter.tolist(),'converged':True}))
                np.savez_compressed(r/'predictions.npz',labels=a[:,12],probabilities=p)
                (r/'RESULT.json').write_text(json.dumps({'utility':summarize(a[:,12],p)}))
            self.assertEqual(check(base/'out',base/'input')['status'],'passed')
            r=base/'out/raw/RESULT.json';data=json.loads(r.read_text());data['utility']['balanced_accuracy']=.99;r.write_text(json.dumps(data))
            with self.assertRaises(ValueError):check(base/'out',base/'input')

if __name__=='__main__':unittest.main(verbosity=2)
