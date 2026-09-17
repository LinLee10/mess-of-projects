import unittest,numpy as np
from audit_recovery import score,compare
class AuditContracts(unittest.TestCase):
 def setUp(self):self.y=np.arange(7);self.p=np.eye(7)
 def test_perfect(self):self.assertEqual(score(self.y,self.p)['balanced_accuracy'],1)
 def test_known_error(self):
  self.p[0]=self.p[1];self.assertEqual(score(self.y,self.p)['correct'],6)
 def test_corrupt_count(self):
  a=score(self.y,self.p)
  with self.assertRaises(ValueError):compare(a,dict(a,correct=6))
 def test_corrupt_recall(self):
  a=score(self.y,self.p)
  with self.assertRaises(ValueError):compare(a,dict(a,class_recall=[0]*7))
 def test_nan(self):
  self.p[0,0]=np.nan
  with self.assertRaises(ValueError):score(self.y,self.p)
 def test_not_normalized(self):
  with self.assertRaises(ValueError):score(self.y,self.p*.5)
 def test_bad_label(self):
  self.y[0]=7
  with self.assertRaises(ValueError):score(self.y,self.p)
 def test_shape(self):
  with self.assertRaises(ValueError):score(self.y,self.p[:6])
if __name__=='__main__':unittest.main()
