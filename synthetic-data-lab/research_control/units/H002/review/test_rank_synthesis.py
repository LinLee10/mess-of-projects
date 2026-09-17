"""Preexecution contracts for exact conditional marginals and empirical beta law."""
import unittest,sys
from pathlib import Path
import numpy as np
from scipy.special import betainc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rank_synthesis import assign_columns,ordinal,generate


def inputs():
    r=np.random.default_rng(4);a=np.zeros((31,13),dtype=np.int64);a[:,:10]=r.integers(0,80,(31,10));a[16:,10:]=[1,3,2];a[-1,10:]=[3,39,6];a[:,4]=5
    c,inv,n=np.unique(a[:,10:],axis=0,return_inverse=True,return_counts=True)
    m={'contexts':c,'counts':n,'offsets':np.r_[0,np.cumsum(n)],'group_indices':np.argsort(inv,kind='stable'),'correlation':np.tile(np.eye(10),(len(n),1,1))}
    return a,m


class Contracts(unittest.TestCase):
    def test_beta_mixture_uniform_identity(self):
        for n in (2,7,31):
            r=np.arange(1,n+1);points=np.array([.001,.1,.5,.9,.999])
            for p in points:self.assertAlmostEqual(np.mean(betainc(r,n+1-r,p)),p,places=12)
    def test_ordinal_ties_are_a_permutation(self):
        np.testing.assert_array_equal(np.sort(ordinal(np.array([1,1,2,3,3]),np.arange(5),0)),np.arange(1,6))
    def test_ordinal_is_identity_keyed(self):
        x=np.array([0,0,1,1]);ids=np.arange(4);p=np.array([3,1,2,0]);a=ordinal(x,ids,1);b=ordinal(x[p],ids[p],1)
        np.testing.assert_array_equal(a[p],b)
    def test_each_context_marginal_identical(self):
        t,m=inputs()
        for method in ('gaussian_rank','beta_rank','beta_wide_rank','independent_rank'):
            a,_=generate(t,m,17,method)
            for c in m['contexts']:
                x=t[np.all(t[:,10:]==c,axis=1)];y=a[np.all(a[:,10:]==c,axis=1)]
                np.testing.assert_array_equal(np.sort(x[:,:10],axis=0),np.sort(y[:,:10],axis=0))
    def test_constant_feature_preserved(self):
        t,m=inputs();a,_=generate(t,m,17,'beta_rank');self.assertTrue((a[:,4]==5).all())
    def test_repeated_generation_exact(self):
        t,m=inputs();a,_=generate(t,m,17,'beta_rank');b,_=generate(t,m,17,'beta_rank');np.testing.assert_array_equal(a,b)
    def test_different_seed_changes_joint_rows(self):
        t,m=inputs();a,_=generate(t,m,17,'beta_rank');b,_=generate(t,m,29,'beta_rank');self.assertFalse(np.array_equal(a,b))
    def test_invalid_mode_rejected(self):
        t,m=inputs()
        with self.assertRaises(ValueError):generate(t,m,17,'unknown')
    def test_invalid_seed_rejected(self):
        t,m=inputs()
        with self.assertRaises(ValueError):generate(t,m,-1,'beta_rank')
    def test_duplicates_in_ids_rejected(self):
        with self.assertRaises(ValueError):ordinal(np.array([1,2]),np.array([0,0]),1)
    def test_nonfinite_latent_rejected(self):
        with self.assertRaises(ValueError):assign_columns(np.array([[np.nan]]),np.array([[1]]))
    def test_assignment_ties_are_identity_invariant(self):
        u=np.zeros((5,2));ref=np.arange(10).reshape(5,2);ids=np.arange(5);p=np.array([3,1,4,0,2])
        a=assign_columns(u,ref,ids);b=assign_columns(u[p],ref,ids[p]);np.testing.assert_array_equal(a[p],b)
    def test_assignment_retains_association_of_latent_ranks(self):
        a=assign_columns(np.array([[3.,1.],[1.,3.],[2.,2.]]),np.array([[10,10],[20,20],[30,30]]))
        np.testing.assert_array_equal(a,[[30,10],[10,30],[20,20]])

if __name__=='__main__':unittest.main()
