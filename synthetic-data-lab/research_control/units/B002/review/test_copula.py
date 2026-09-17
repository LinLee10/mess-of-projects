"""Tests specify the baseline contract before experimental fitting."""
import sys, unittest, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from copula import ContextCopula


def fixture():
    rng=np.random.default_rng(31)
    a=np.zeros((101,13),dtype=np.int64)
    a[:,:10]=rng.integers(1,70,(101,10));a[:,0]=np.arange(101)+1000
    a[50:,10]=1;a[50:,11]=2;a[50:,12]=1
    a[100,10:]=[3,39,6]
    a[:,3]=4
    return a


class Contract(unittest.TestCase):
    def test_preserves_all_context_counts(self):
        a=fixture(); b,_=ContextCopula().fit(a).draw(17)
        np.testing.assert_array_equal(np.unique(a[:,10:],axis=0,return_counts=True)[1],np.unique(b[:,10:],axis=0,return_counts=True)[1])
        self.assertEqual(b.shape,a.shape)
    def test_draws_stay_inside_context_feature_support(self):
        a=fixture();m=ContextCopula().fit(a);b,_=m.draw(17)
        for c in m.contexts:
            x=a[np.all(a[:,10:]==c,axis=1),:10]; y=b[np.all(b[:,10:]==c,axis=1),:10]
            self.assertTrue((y>=x.min(0)).all() and (y<=x.max(0)).all())
    def test_constant_column_is_exact(self):
        b,_=ContextCopula().fit(fixture()).draw(17);self.assertTrue((b[:,3]==4).all())
    def test_singleton_context_is_disclosed_copy(self):
        a=fixture();b,_=ContextCopula().fit(a).draw(17)
        np.testing.assert_array_equal(b[b[:,12]==6],a[a[:,12]==6])
    def test_same_seed_roundtrip_is_exact(self):
        m=ContextCopula().fit(fixture()); a,_=m.draw(17)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.npz';m.save(p);b,_=ContextCopula.load(p).draw(17)
            np.testing.assert_array_equal(a,b)
    def test_different_seed_changes_draw(self):
        m=ContextCopula().fit(fixture());a,_=m.draw(17);b,_=m.draw(18);self.assertFalse(np.array_equal(a,b))
    def test_covariance_is_positive_definite(self):
        m=ContextCopula().fit(fixture()); self.assertTrue((np.linalg.eigvalsh(m.correlation)>0).all())
        np.testing.assert_allclose(np.diagonal(m.correlation,axis1=1,axis2=2),1,atol=1e-12)
    def test_fitted_distribution_is_order_invariant(self):
        a=fixture();m=ContextCopula().fit(a);n=ContextCopula().fit(a[::-1])
        np.testing.assert_array_equal(m.sorted,n.sorted);np.testing.assert_allclose(m.correlation,n.correlation,atol=1e-12)
    def test_malformed_schema_rejected(self):
        with self.assertRaises(ValueError):ContextCopula().fit(fixture()[:,:12])
    def test_fractional_table_rejected(self):
        with self.assertRaises(ValueError):ContextCopula().fit(fixture().astype(float))
    def test_unknown_context_rejected(self):
        a=fixture();a[0,11]=99
        with self.assertRaises(ValueError):ContextCopula().fit(a)
    def test_negative_seed_rejected(self):
        with self.assertRaises(ValueError):ContextCopula().fit(fixture()).draw(-1)
    def test_fit_uses_every_training_row(self):
        a=fixture();m=ContextCopula().fit(a)
        np.testing.assert_array_equal(np.sort(m.group_indices),np.arange(len(a)))
        for g in range(len(m.counts)):
            sl=slice(m.offsets[g],m.offsets[g+1]);np.testing.assert_array_equal(m.sorted[sl],np.sort(a[m.group_indices[sl],:10],axis=0))

if __name__=='__main__':unittest.main()
