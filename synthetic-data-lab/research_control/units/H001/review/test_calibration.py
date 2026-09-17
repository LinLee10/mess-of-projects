"""R002 acceptance controls written without choosing a preferred model score."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibration import rank_map,hard_valid,confusion,benchmark,calibrate


def fixture(n=700):
    r=np.random.default_rng(101);a=np.zeros((n,13),dtype=np.int64)
    a[:,0]=r.integers(1500,4000,n);a[:,1]=r.integers(0,361,n);a[:,2]=r.integers(0,70,n)
    a[:,3]=r.integers(0,1000,n);a[:,4]=r.integers(-100,100,n);a[:,5]=r.integers(0,6000,n)
    a[:,6:9]=r.integers(0,256,(n,3));a[:,9]=r.integers(0,6000,n);a[:,10]=r.integers(0,4,n);a[:,11]=r.integers(0,40,n);a[:,12]=np.arange(n)%7
    return a


class ContractTests(unittest.TestCase):
    def test_correct_confusion(self):
        x=confusion(np.array([0,0,0,1]),np.array([0,0,1,1]),k=2)
        self.assertEqual(x['correct'],3);self.assertAlmostEqual(x['balanced_accuracy'],5/6)
    def test_fractional_rejected(self):
        with self.assertRaises(ValueError):confusion(np.array([0.,1.2]),np.array([0,1]),k=2)
    def test_valid_realistic_fixture(self):self.assertTrue(hard_valid(fixture()).all())
    def test_invalid_slope(self):
        a=fixture();a[0,2]=91;self.assertFalse(hard_valid(a)[0])
    def test_signed_vertical_distance_permitted(self):
        a=fixture();a[0,4]=-100;self.assertTrue(hard_valid(a)[0])
    def test_category_rejected(self):
        a=fixture();a[0,10]=4;self.assertFalse(hard_valid(a)[0])
    def test_keyed_permutation_equivariance(self):
        r=np.random.default_rng(22);x=r.integers(0,8,700);t=np.arange(1200);ids=r.permutation(700);p=r.permutation(700)
        u=rank_map(x,t,ids);v=rank_map(x[p],t,ids[p]);self.assertTrue(np.array_equal(u[p],v))
    def test_no_ties_legacy_equals_keyed(self):
        x=np.arange(700)[::-1];ids=np.arange(700);self.assertTrue(np.array_equal(rank_map(x,x,ids,mode='legacy'),rank_map(x,x,ids)))
    def test_legacy_order_defect_reproduced(self):
        x=np.zeros(700);ids=np.arange(700);p=np.random.default_rng(9).permutation(700)
        self.assertFalse(np.array_equal(rank_map(x,ids,ids,mode='legacy')[p],rank_map(x[p],ids,ids[p],mode='legacy')))
    def test_constant_midrank_remains_constant(self):
        x=rank_map(np.zeros(700),np.arange(700),np.arange(700),mode='midrank');self.assertEqual(len(np.unique(x)),1)
    def test_keyed_does_not_synchronize_constant_columns(self):
        x=np.zeros(2000);ids=np.arange(2000)
        a=rank_map(x,ids,ids,column=0);b=rank_map(x,ids,ids,column=1)
        self.assertLess(abs(np.corrcoef(a,b)[0,1]),.1)
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):rank_map(np.arange(4),np.arange(4),np.array([0,0,2,3]))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):rank_map(np.array([np.nan]),np.array([1]),np.array([0]))
    def test_copy_control_caught(self):
        a=fixture();m=benchmark(a,a,a);self.assertEqual(m['copy_count'],len(a));self.assertEqual(m['mean_numeric_ks'],0)
    def test_collapse_control_caught(self):
        a=fixture();b=np.repeat(a[:7],100,axis=0);m=benchmark(b,a,a);self.assertEqual(m['unique_count'],7)
    def test_label_shuffle_fails_conditional_not_marginal(self):
        a=fixture();a[:,0]=2000+100*a[:,12];b=a.copy();b[:,12]=np.random.default_rng(1).permutation(b[:,12]);m=benchmark(b,a,a)
        self.assertEqual(m['mean_numeric_ks'],0);self.assertGreater(m['class_balanced_conditional_ks'],.04)
    def test_tail_erasure_detected(self):
        a=fixture();b=a.copy()
        for j in range(10):b[:,j]=np.clip(b[:,j],*np.quantile(a[:,j],[.1,.9]))
        self.assertGreater(benchmark(b,a,a)['tail_mean_abs_error'],0)
    def test_calibration_preserves_context_columns(self):
        a=fixture();b=calibrate(a,a,np.arange(len(a)),context=True);self.assertTrue(np.array_equal(a[:,10:],b[:,10:]))


if __name__=='__main__':unittest.main(verbosity=2)
