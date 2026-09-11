import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from audit_v2 import AuditFailure, canonical, digest, independent_metrics, verify_manifest, safe, compare_metrics

class IndependentCheckerTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        s = independent_metrics(np.array([0,0,0,1]), np.array([[.8,.2],[.7,.3],[.1,.9],[.2,.8]]))
        self.assertEqual(s['confusion_matrix'], [[2,1],[0,1]])
        self.assertAlmostEqual(s['accuracy'], .75)
        self.assertAlmostEqual(s['balanced_accuracy'], 5/6)
        self.assertAlmostEqual(s['macro_f1'], (.8+2/3)/2)
        self.assertAlmostEqual(s['log_loss'], -sum(map(math.log,[.8,.7,.1,.8]))/4)
    def test_wrong_labels_do_not_pass(self):
        s=independent_metrics(np.array([0,1]), np.array([[.01,.99],[.99,.01]]))
        self.assertEqual(s['balanced_accuracy'],0.0)
    def test_fractional_and_boolean_labels_rejected(self):
        for y in [np.array([0.,1.9]),np.array([False,True])]:
            with self.subTest(y=y), self.assertRaises(AuditFailure):
                independent_metrics(y,np.array([[.9,.1],[.1,.9]]))
    def test_invalid_probabilities_rejected(self):
        for p in [np.array([[np.nan,0],[.1,.9]]),np.array([[.4,.4],[.1,.9]]),np.array([[-.1,1.1],[.1,.9]])]:
            with self.subTest(p=p), self.assertRaises(AuditFailure):
                independent_metrics(np.array([0,1]),p)
    def test_unknown_class_rejected(self):
        with self.assertRaises(AuditFailure):
            independent_metrics(np.array([0,2]),np.array([[.9,.1],[.1,.9]]))
    def test_metric_tampering_rejected(self):
        original=independent_metrics(np.array([0,1]),np.array([[.9,.1],[.1,.9]]))
        changed=dict(original,balanced_accuracy=.5)
        with self.assertRaises(AuditFailure):compare_metrics(original,changed,'controlled mutation')
    def manifest(self,root):
        (root/'data.txt').write_text('original')
        m={'schema_version':1,'files':{'data.txt':digest(root/'data.txt')}}
        m['manifest_sha256']=canonical(m)
        (root/'FROZEN.json').write_text(json.dumps(m))
    def test_modified_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.manifest(p);verify_manifest(p,('data.txt',))
            (p/'data.txt').write_text('modified')
            with self.assertRaises(AuditFailure):verify_manifest(p)
    def test_missing_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.manifest(p)
            with self.assertRaises(AuditFailure):verify_manifest(p,('required.txt',))
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AuditFailure):safe(Path(d),'../outside.txt')
    def test_empty_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m={'schema_version':1,'files':{}};m['manifest_sha256']=canonical(m)
            (p/'FROZEN.json').write_text(json.dumps(m))
            with self.assertRaises(AuditFailure):verify_manifest(p)

if __name__=='__main__':unittest.main()
