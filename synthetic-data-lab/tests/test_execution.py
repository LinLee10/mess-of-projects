"""Additional protocol and export tests; assertions do not encode a winning method."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from synthlab_next.common import seal,verify_seal
from synthlab_next.generators import CTGAN,TVAE,TabDDPM,Copula,Interpolation,GeneratorConfig,Denoiser
from synthlab_next.inference import load_generator,sample_csv
from synthlab_next.experiment import evaluate_experiment,labels_for,subset


class ExecutionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);r=np.random.default_rng(91)
        self.y=np.repeat([0,1],24);self.x=r.normal(size=(48,3))+.8*self.y[:,None]

    def test_all_saved_generators_reproduce_draws(self):
        for kind in (CTGAN,TVAE,TabDDPM,Copula,Interpolation):
            with self.subTest(kind=kind.__name__),tempfile.TemporaryDirectory() as tmp:
                model=(kind(GeneratorConfig(steps=30,width=16,latent=8,batch_size=20,pac=2,diffusion_steps=8),7)
                    if kind in (CTGAN,TVAE,TabDDPM) else kind(7)).fit(self.x,self.y)
                model.save(tmp);loaded=load_generator(tmp)
                if kind in (CTGAN,TVAE):
                    a,b=model.sample_native(16,seed=13);c,d=loaded.sample_native(16,seed=13)
                else:
                    a,b=model.sample_labels(np.array([0,1]*8),seed=13);c,d=loaded.sample_labels(np.array([0,1]*8),seed=13)
                np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)

    def test_unconditional_ablation_ignores_class_input(self):
        torch.manual_seed(42);net=Denoiser(3,2,16,False);x=torch.randn(5,3);t=torch.arange(5)
        torch.testing.assert_close(net(x,t,torch.zeros(5,dtype=torch.long)),net(x,t,torch.ones(5,dtype=torch.long)),rtol=0,atol=0)

    def test_conditioned_denoiser_changes_with_class(self):
        torch.manual_seed(42);net=Denoiser(3,2,16,True);x=torch.randn(5,3);t=torch.arange(5)
        self.assertFalse(torch.equal(net(x,t,torch.zeros(5,dtype=torch.long)),net(x,t,torch.ones(5,dtype=torch.long))))

    def test_evaluation_refuses_before_loading_test(self):
        with tempfile.TemporaryDirectory() as tmp,patch('numpy.load',side_effect=AssertionError('Test touched')):
            with self.assertRaises(ValueError):evaluate_experiment(tmp)

    def test_changed_candidate_refuses_before_loading_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'candidate').write_text('original');seal(p,['candidate']);(p/'candidate').write_text('changed')
            with patch('numpy.load',side_effect=AssertionError('Test touched')):
                with self.assertRaises(ValueError):evaluate_experiment(p)

    def test_matched_class_counts_and_no_sampling_substitution(self):
        labels=labels_for(self.y,20,7);a,b=subset(self.x,self.y,labels,11)
        self.assertEqual(np.bincount(b).tolist(),[10,10]);self.assertEqual(len(np.unique(a,axis=0)),20)
        for row,label in zip(a,b):self.assertTrue(any(np.array_equal(row,r) for r in self.x[self.y==label]))

    def test_csv_reload_export_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Copula(17).fit(self.x,self.y).save(root/'generator')
            report=sample_csv(root/'generator',root/'sample.csv',12,23)
            self.assertEqual(report['rows'],12);self.assertEqual(len((root/'sample.csv').read_text().splitlines()),13)
            with self.assertRaises(ValueError):sample_csv(root/'generator',root/'sample.csv',12,23)

if __name__=='__main__':unittest.main()
