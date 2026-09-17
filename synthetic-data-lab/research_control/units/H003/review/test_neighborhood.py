import unittest,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root.parent/'H002/src')]
from neighborhood import fit_geometry,draw


def source():
    rng=np.random.default_rng(19);t=rng.integers(1,40,(101,13));t[:,10:]=0;t[51:100,10:]=[1,2,3];t[100,10:]=[3,39,6];t[:,3]=7
    c,inv,count=np.unique(t[:,10:],axis=0,return_inverse=True,return_counts=True)
    m={'contexts':c,'counts':count,'offsets':np.r_[0,np.cumsum(count)],'group_indices':np.argsort(inv,kind='stable')}
    return t,m


class Tests(unittest.TestCase):
    def test_full_training_coverage(self):
        t,m=source();g=fit_geometry(t,m);self.assertEqual(g['neighbors'].shape,(101,5));self.assertEqual(len(g['loo_distance']),101)
    def test_nearest_other_matches_brute_force(self):
        t,m=source();g=fit_geometry(t,m)
        for i in range(100):
            ids=np.flatnonzero(np.all(t[:,10:]==t[i,10:],axis=1));ids=ids[ids!=i]
            d=cdist(t[i,None,:10]/g['scale'],t[ids,:10]/g['scale'])[0]
            self.assertAlmostEqual(d.min(),g['loo_distance'][i],places=12)
    def test_no_self_neighbor_except_singleton(self):
        t,m=source();g=fit_geometry(t,m);self.assertFalse((g['neighbors'][:100]==np.arange(100)[:,None]).any());self.assertTrue((g['neighbors'][100]==100).all())
    def test_contexts_do_not_mix(self):
        t,m=source();g=fit_geometry(t,m)
        for i in range(101):self.assertTrue((t[g['neighbors'][i],10:]==t[i,10:]).all())
    def test_finite_positive_bounded_widths(self):
        t,m=source();g=fit_geometry(t,m);self.assertTrue(np.isfinite(g['bandwidth']).all());self.assertTrue((g['bandwidth']>=1).all());self.assertTrue((g['bandwidth']<=m['counts']).all())
    def test_exact_context_marginals(self):
        t,m=source();g=fit_geometry(t,m)
        for method in ('neighbor_rank','adaptive_beta'):
            a,_=draw(t,m,g,17,method)
            for c in m['contexts']:
                x=t[np.all(t[:,10:]==c,axis=1),:10];y=a[np.all(a[:,10:]==c,axis=1),:10];np.testing.assert_array_equal(np.sort(x,axis=0),np.sort(y,axis=0))
    def test_raw_interpolation_formula(self):
        t,m=source();g=fit_geometry(t,m);a,p=draw(t,m,g,17,'neighbor_raw');w=p['weight'][:,None]
        np.testing.assert_array_equal(a[:,:10],np.rint((1-w)*t[p['anchor'],:10]+w*t[p['other'],:10]))
    def test_replay_exact(self):
        t,m=source();g=fit_geometry(t,m)
        for method in ('neighbor_raw','neighbor_rank','adaptive_beta'):
            a,_=draw(t,m,g,17,method);b,_=draw(t,m,g,17,method);np.testing.assert_array_equal(a,b)
    def test_constant_features_remain(self):
        t,m=source();g=fit_geometry(t,m);a,_=draw(t,m,g,17,'adaptive_beta');self.assertTrue((a[:,3]==7).all())
    def test_constant_group_is_not_called_private(self):
        t=np.zeros((9,13),dtype=int);m={'contexts':np.zeros((1,3),int),'counts':np.array([9]),'offsets':np.array([0,9]),'group_indices':np.arange(9)}
        g=fit_geometry(t,m);a,_=draw(t,m,g,17,'adaptive_beta');np.testing.assert_array_equal(a,t)
    def test_invalid_method_refused(self):
        t,m=source();g=fit_geometry(t,m)
        with self.assertRaises(ValueError):draw(t,m,g,17,'unknown')
    def test_fit_repeated_exact(self):
        t,m=source();g=fit_geometry(t,m);h=fit_geometry(t,m)
        for k in g:np.testing.assert_array_equal(g[k],h[k])

if __name__=='__main__':unittest.main()
