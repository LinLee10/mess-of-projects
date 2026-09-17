import unittest,numpy as np
from check_h003 import keyed_rank_assign,ordinal,require

class CheckerControls(unittest.TestCase):
    def test_hand_assignment(self):
        a=keyed_rank_assign(np.array([[3.,1.],[1.,3.],[2.,2.]]),np.array([[10,10],[20,20],[30,30]]),np.arange(3))
        np.testing.assert_array_equal(a,[[30,10],[10,30],[20,20]])
    def test_order_invariant_ties(self):
        u=np.zeros((5,2));x=np.arange(10).reshape(5,2);ids=np.arange(5);p=np.array([2,4,1,3,0])
        a=keyed_rank_assign(u,x,ids);b=keyed_rank_assign(u[p],x,ids[p]);np.testing.assert_array_equal(a[p],b)
    def test_ordinal_recovers_all_ranks(self):
        x=np.tile(np.array([5,1,5,3])[:,None],(1,10));r=ordinal(x,np.arange(4))
        for j in range(10):np.testing.assert_array_equal(np.sort(r[:,j]),[1,2,3,4])
    def test_ordinal_respects_strict_order(self):
        x=np.tile(np.array([5,1,7,3])[:,None],(1,10));r=ordinal(x,np.arange(4));np.testing.assert_array_equal(r[:,0],[3,1,4,2])
    def test_failed_gate_rejects(self):
        with self.assertRaises(ValueError):require(False,'controlled failure')
    def test_correct_gate_accepts(self):require(True,'pass')

if __name__=='__main__':unittest.main()
