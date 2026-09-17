import unittest,numpy as np
from context_proximity import distances_summary
class DistanceControls(unittest.TestCase):
 def test_identical(self):
  x=np.arange(1,101);a=distances_summary(x,x);self.assertEqual(a['ks_distance'],0);self.assertEqual(a['synthetic_below_reference_quantiles'],a['reference_below_own_quantiles'])
 def test_copies(self):self.assertEqual(distances_summary(np.zeros(100),np.arange(1,101))['synthetic_below_reference_quantiles'][1],1)
 def test_pooled_can_hide_context(self):
  low=np.arange(1,101);high=np.arange(201,301)
  self.assertEqual(distances_summary(np.r_[low,high],np.r_[high,low])['ks_distance'],0)
  self.assertEqual(distances_summary(low,high)['ks_distance'],1)
 def test_nan_rejected(self):
  with self.assertRaises(ValueError):distances_summary([np.nan],[1])
 def test_empty_rejected(self):
  with self.assertRaises(ValueError):distances_summary([], [1])
 def test_quantiles(self):
  a=distances_summary(np.arange(1,101),np.arange(1,101));self.assertTrue(all(np.diff(a['reference_quantiles'])>=0));self.assertEqual(a['synthetic_below_reference_quantiles'][3],.5)
if __name__=='__main__':unittest.main()
