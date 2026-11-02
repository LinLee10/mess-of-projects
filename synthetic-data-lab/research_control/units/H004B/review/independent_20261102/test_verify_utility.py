"""Small independent controls. No production modules or real outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import verify_utility as v


class Controls(unittest.TestCase):
    def setUp(self):
        self.y=np.arange(7,dtype=np.int64)
        self.p=np.eye(7)

    def test_01_exact_arithmetic(self):
        s=v.score(self.y,self.p)
        self.assertEqual(s['correct'],7)
        self.assertEqual(s['confusion_matrix'],np.eye(7,dtype=int).tolist())
        self.assertEqual(s['balanced_accuracy'],1.)

    def test_02_asymmetric_confusion(self):
        y=np.r_[self.y,0,0]; p=np.eye(7)[np.r_[self.y,1,1]]
        s=v.score(y,p)
        self.assertEqual(s['correct'],7)
        self.assertEqual(s['confusion_matrix'][0][1],2)
        self.assertAlmostEqual(s['balanced_accuracy'],(6+1/3)/7)
        self.assertAlmostEqual(s['accuracy'],7/9)

    def test_03_nonfinite_probability(self):
        for value in (np.nan,np.inf,-np.inf):
            p=self.p.copy();p[0,0]=value
            with self.assertRaises(ValueError):v.score(self.y,p)

    def test_04_probability_shape_and_range(self):
        for p in (self.p[:,:6], self.p[:,:,None], self.p*2, self.p-.01):
            with self.assertRaises(ValueError):v.score(self.y,p)

    def test_05_normalization(self):
        p=self.p*.99
        with self.assertRaises(ValueError):v.score(self.y,p)

    def test_06_label_corruption(self):
        for y in (self.y.astype(float),self.y[:,None],self.y+1,self.y[:-1]):
            with self.assertRaises(ValueError):v.score(y,self.p)

    def test_07_absent_class(self):
        with self.assertRaises(ValueError):v.score(self.y[:-1],self.p[:-1])

    def test_08_saved_arithmetic_corruption(self):
        expected=v.score(self.y,self.p)
        for key in expected:
            bad=copy.deepcopy(expected)
            if key=='confusion_matrix':bad[key][0][0]=0
            elif key=='class_recall':bad[key][0]=.9
            else:bad[key]=bad[key]+1
            with self.assertRaises(ValueError):v.equal(bad,expected)

    def test_09_reload_tolerance_and_classes(self):
        v.compare_inference(self.p,self.p.copy())
        with self.assertRaises(ValueError):v.compare_inference(self.p,np.roll(self.p,1,axis=1))
        a=np.full((7,7),.1);a[:,0]=.25;a[:,1]=.25
        b=a.copy();b[:,0]-=1e-12;b[:,1]+=1e-12
        with self.assertRaisesRegex(ValueError,'classes changed'):v.compare_inference(a,b)

    def test_10_matrix_coverage(self):
        rows=[dict(method=m,seed=s) for m,s in sorted(v.CONDITIONS)]
        self.assertEqual(len(v.matrix(rows)),19)
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):v.matrix(bad)

    def screen_rows(self):
        return [dict(method=m,seed=s,tree=dict(balanced_accuracy=.5,class_recall=[.5]*7),linear=dict(balanced_accuracy=.5,class_recall=[.5]*7)) for m,s in sorted(v.CONDITIONS)]

    def test_11_screen_class_veto(self):
        rows=self.screen_rows();g=dict(empirical_median=True,empirical_quantile=True)
        self.assertTrue(v.engineering(rows,g)['empirical_median']['exploratory_engineering_screen_pass'])
        for row in rows:
            if row['method']=='empirical_median':row['linear']['class_recall'][6]=.469
        self.assertFalse(v.engineering(rows,g)['empirical_median']['exploratory_engineering_screen_pass'])

    def test_12_screen_balanced_and_geometry_veto(self):
        for learner,loss in (('tree',.011),('linear',.006)):
            rows=self.screen_rows()
            for r in rows:
                if r['method']=='empirical_quantile':r[learner]['balanced_accuracy']-=loss
            self.assertFalse(v.engineering(rows,dict(empirical_median=True,empirical_quantile=True))['empirical_quantile']['exploratory_engineering_screen_pass'])
        self.assertFalse(v.engineering(self.screen_rows(),dict(empirical_median=False,empirical_quantile=True))['empirical_median']['exploratory_engineering_screen_pass'])

    def test_13_manifest_hash_coverage_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);f=root/'a';f.write_text('a')
            def put(d): (root/'M.json').write_text(json.dumps(d))
            put({'a':v.sha(f)});v.manifest(root,'M.json',('M.json',),True)
            f.write_text('b')
            with self.assertRaises(ValueError):v.manifest(root,'M.json')
            put({'../a':'x'})
            with self.assertRaises(ValueError):v.manifest(root,'M.json')
            put({'a':v.sha(f)});(root/'extra').write_text('x')
            with self.assertRaises(ValueError):v.manifest(root,'M.json',('M.json',),True)

    def test_14_duplicate_json_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x'
            for text in ('{"a":1,"a":2}','{"a":NaN}'):
                p.write_text(text)
                with self.assertRaises(ValueError):v.read(p)

    def model(self):
        t=np.zeros((7,13),dtype=np.int64);t[:,10]=np.arange(7)%4;t[:,11]=np.arange(7);t[:,12]=np.arange(7)
        m=dict(mean=[0.]*10,scale=[1e-8]*10,coef=np.zeros((7,54)).tolist(),intercept=[0.]*7,classes=list(range(7)),configuration=dict(C=1.,solver='lbfgs',max_iter=1000,tol=1e-5),iterations=[2],converged=True,warnings=[])
        return t,m

    def test_15_linear_encoding_inference(self):
        t,m=self.model();m['coef'][0][10]=2.;m['coef'][1][14+3]=3.
        p=v.linear_inference(m,t,t)
        self.assertEqual(p[0].argmax(),0)
        self.assertEqual(p[3].argmax(),1)
        np.testing.assert_allclose(p.sum(1),1)

    def test_16_linear_corruptions(self):
        t,base=self.model()
        for key,value in (('classes',list(range(6))),('converged',False),('coef',[[0.]*54]*6),('scale',[0.]*10),('mean',[1.]*10),('iterations',[1000]),('intercept',[float('nan')]*7)):
            m=copy.deepcopy(base);m[key]=value
            with self.assertRaises(ValueError):v.linear_inference(m,t,t)

    def test_17_reference_lineage_and_marginals(self):
        n=7;t=np.zeros((n,13),dtype=np.int64);t[:,:10]=np.arange(n)[:,None]%3
        groups=dict(contexts=np.zeros((1,3),dtype=np.int64),counts=np.array([n]),offsets=np.array([0,n]),group_indices=np.arange(n))
        # Fixed independently calculated identity hash vector guards the tie rule.
        self.assertEqual(int(v.tie_keys(np.array([0]),0)[0]),7073115177516498683)
        r=np.empty((n,10),dtype=np.int64)
        for j in range(10):r[np.lexsort((v.tie_keys(np.arange(n),j),t[:,j])),j]=np.arange(1,n+1)
        rng=np.random.default_rng(np.random.SeedSequence([43017,0,3202]));anchors=rng.integers(0,n,size=n);u=rng.beta(r[anchors],n+1-r[anchors])
        inv=t.copy();rank=t.copy();sx=np.sort(t[:,:10],axis=0)
        inv[:,:10]=sx[np.minimum((n*u).astype(int),n-1),np.arange(10)]
        for j in range(10):rank[np.lexsort((v.tie_keys(np.arange(n),j),u[:,j])),j]=sx[:,j]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            np.savez(root/'paper_lineage_43017.npz',latent=u,anchor_ids=anchors,original_ids=np.arange(n),contexts=groups['contexts'],counts=groups['counts'])
            for m,a in (('iid_beta_inverse',inv),('iid_beta_rank',rank)):np.savez(root/f'{m}_43017.npz',table=a,contexts=groups['contexts'],counts=groups['counts'])
            v.references(t,groups,43017,root)
            np.testing.assert_array_equal(np.sort(rank[:,:10],axis=0),sx)
            inv[0,0]+=1
            np.savez(root/'iid_beta_inverse_43017.npz',table=inv,contexts=groups['contexts'],counts=groups['counts'])
            with self.assertRaises(ValueError):v.references(t,groups,43017,root)

if __name__=='__main__':unittest.main()
