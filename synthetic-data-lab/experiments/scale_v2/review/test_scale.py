"""Independent contract expectations fixed before implementing the scale runner.
Hand constructed cases are software controls, never research observations.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from coverage_v2 import CoverageCycle, load_coverage_generators
from fullscale import group_partition, project_geometry, geometry_errors, metrics, compare_selection, full_student

class ScaleContracts(unittest.TestCase):
    def test_cycle_covers_every_row_before_repeating(self):
        c=CoverageCycle(19,7)
        got=np.r_[c.take(8),c.take(8),c.take(3)]
        self.assertEqual(set(got.tolist()),set(range(19)))
        self.assertEqual(c.counts.tolist(),[1]*19)
    def test_cycle_is_repeatable(self):
        a,b=CoverageCycle(23,71),CoverageCycle(23,71)
        np.testing.assert_array_equal(a.take(65),b.take(65))
    def test_cycle_refuses_invalid_sizes(self):
        for n in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):CoverageCycle(n,17)
    def test_partition_has_no_cap_or_unused_rows(self):
        x=np.arange(400,dtype=float).reshape(200,2)
        parts=group_partition(x)
        self.assertEqual(sum(map(len,parts.values())),200)
        self.assertEqual(set(parts),{'train','dev','test'})
        ids=np.concatenate(list(parts.values()))
        self.assertEqual(set(ids.tolist()),set(range(200)))
    def test_identical_features_remain_in_same_split_even_with_different_labels(self):
        x=np.array([[1.,2.],[3.,4.],[1.,2.],[7.,8.],[3.,4.]])
        p=group_partition(x);where={i:k for k,ids in p.items() for i in ids}
        self.assertEqual(where[0],where[2]);self.assertEqual(where[1],where[4])
    def test_partition_is_invariant_to_row_order(self):
        x=np.arange(600,dtype=float).reshape(200,3);order=np.arange(199,-1,-1)
        a,b=group_partition(x),group_partition(x[order]);
        for k in a:self.assertEqual(set(a[k].tolist()),set(order[b[k]].tolist()))
    def test_geometry_reconstruction_not_class_label_repair(self):
        names=['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness']
        x=np.array([[100.,40.,15.,10.,9.,9.,120.,9.,.8,9.,9.,9.]])
        fixed=project_geometry(x,names)
        self.assertAlmostEqual(fixed[0,4],1.5)
        self.assertAlmostEqual(fixed[0,7],np.sqrt(400/np.pi))
        self.assertAlmostEqual(fixed[0,9],100/120)
        self.assertGreater(geometry_errors(x,names)['any_equation_violation_rate'],0)
        self.assertEqual(geometry_errors(fixed,names)['any_equation_violation_rate'],0)
        np.testing.assert_array_equal(fixed[0,[0,1,2,3,6,8]],x[0,[0,1,2,3,6,8]])
    def test_geometry_does_not_disguise_invalid_primitives(self):
        names=['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness']
        x=np.array([[-1.,40.,3.,10.,1.,1.,120.,1.,.8,1.,1.,1.]])
        with self.assertRaises(ValueError):project_geometry(x,names)
    def test_metrics_known_outcomes(self):
        m=metrics(np.array([0,0,0,1]),np.array([[.8,.2],[.7,.3],[.1,.9],[.2,.8]]))
        self.assertEqual(m['correct'],3);self.assertEqual(m['n'],4)
        self.assertAlmostEqual(m['balanced_accuracy'],5/6)
    def test_probabilities_and_labels_fail_closed(self):
        for y,p in [(np.array([0.,1.9]),np.array([[.9,.1],[.1,.9]])),(np.array([0,1]),np.array([[.9,.9],[.1,.9]]))]:
            with self.assertRaises(ValueError):metrics(y,p)
    def test_selection_uses_dev_only(self):
        rows=[{'method':'a','dev':{'balanced_accuracy':.8,'log_loss':.3},'test':{'balanced_accuracy':0}}, {'method':'b','dev':{'balanced_accuracy':.7,'log_loss':.1},'test':{'balanced_accuracy':1}}]
        self.assertEqual(compare_selection(rows),'a')
        rows[0]['test']['balanced_accuracy']=1;rows[1]['test']['balanced_accuracy']=0
        self.assertEqual(compare_selection(rows),'a')
    def test_student_uses_all_rows_and_roundtrips(self):
        import torch
        torch.set_num_threads(1)
        x=np.column_stack([np.arange(40)/40,np.arange(40)%3]);y=np.arange(40)%2
        with tempfile.TemporaryDirectory() as d:
            p,a=full_student(x,y,x,seed=7,steps=4,batch=16,path=Path(d),return_model=True)
            self.assertEqual(a['unique_training_rows_seen'],40)
            self.assertGreaterEqual(a['minimum_row_exposures'],1)
            np.testing.assert_allclose(p.predict_proba(x).sum(1),1,atol=1e-6)
    def test_neural_generators_cover_all_rows(self):
        import torch
        torch.set_num_threads(1)
        m=load_coverage_generators()
        x=np.column_stack([np.arange(30)/30,np.sin(np.arange(30))]);y=np.arange(30)%2
        for cls in [m.CTGAN,m.TVAE,m.TabDDPM]:
            cfg=m.GeneratorConfig(steps=40,batch_size=10,width=8,latent=4,max_modes=1,pac=1,diffusion_steps=4)
            fitted=cls(cfg,13).fit(x,y)
            self.assertEqual(fitted.audit['unique_training_rows_seen'],30)
            self.assertGreaterEqual(fitted.audit['minimum_row_exposures'],1)
            self.assertNotEqual(fitted.audit['initial_sha256'],fitted.audit['final_sha256'])

if __name__=='__main__':unittest.main()
