import unittest
import numpy as np
from verify_full_tables import ecdf_distance, exact_counts, valid_mask, close, require

class FullTableCheckerTests(unittest.TestCase):
    def test_ecdf_equal(self):self.assertEqual(ecdf_distance(np.array([0,1,1]),np.array([0,1,1])),0)
    def test_ecdf_shift(self):self.assertEqual(ecdf_distance(np.array([0,0]),np.array([1,1])),1)
    def test_ecdf_ties(self):self.assertAlmostEqual(ecdf_distance(np.array([0,0,0,1]),np.array([0,1,1,1])),.5)
    def test_full_records_not_individual_features(self):
        a=np.array([[1,2],[1,2],[2,1]],dtype=np.int64);t=np.array([[1,1],[1,2]],dtype=np.int64)
        self.assertEqual(exact_counts(a,t),(2,2))
    def test_metric_tampering(self):
        with self.assertRaises(ValueError):close(.7,.71,'score')
    def test_nan_not_equal(self):
        with self.assertRaises(ValueError):close(np.nan,np.nan,'score')
    def test_illegal_categories(self):
        a=np.zeros((2,13));a[1,11]=40;self.assertEqual(valid_mask(a).tolist(),[True,False])
    def test_signed_vertical_distance(self):
        a=np.zeros((1,13));a[0,4]=-30;self.assertTrue(valid_mask(a)[0])
    def test_fractional_record(self):
        a=np.zeros((1,13));a[0,1]=.5;self.assertFalse(valid_mask(a)[0])

if __name__=='__main__':unittest.main(verbosity=2)
