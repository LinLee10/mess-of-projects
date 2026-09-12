"""Requirements based tests written before implementation.

The expected values come from algebra and API contracts, not measured winners.
Same author as implementation: this is not an independent external review.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from synthlab_next.common import validate_xy, array_hash, state_hash, write_json, seal, verify_seal
from synthlab_next.data import make_split
from synthlab_next.transforms import ModeTransformer
from synthlab_next.generators import (
    GeneratorConfig, CTGAN, TVAE, TabDDPM, Copula, Interpolation,
    PackedCritic, diffusion_schedule, posterior_from_x0,
)
from synthlab_next.students import StudentConfig, train_student, load_student
from synthlab_next.portfolio import utility_weights, support_mask, select_portfolio


class DataContracts(unittest.TestCase):
    def test_nonfinite_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                validate_xy(np.array([[value], [1.0]]), np.array([0, 1]))

    def test_noninteger_labels_rejected(self):
        with self.assertRaises(ValueError):
            validate_xy(np.ones((3, 2)), np.array([0.1, 1.0, 0.0]))

    def test_wrong_shape_and_missing_class_rejected(self):
        for x, y in [(np.ones(3), np.array([0, 1, 0])),
                     (np.ones((3, 2)), np.array([0, 1])),
                     (np.ones((3, 2)), np.array([0, 2, 2]))]:
            with self.assertRaises(ValueError): validate_xy(x, y)

    def test_split_is_disjoint_and_immutable(self):
        s = make_split('wine', 17)
        ids = [set(s[k]['ids'].tolist()) for k in ('train', 'dev', 'test')]
        self.assertFalse(ids[0] & ids[1]); self.assertFalse(ids[0] & ids[2]); self.assertFalse(ids[1] & ids[2])
        with self.assertRaises(ValueError): s['train']['x'][0, 0] = 999
        again = make_split('wine', 17)
        self.assertEqual(array_hash(s['train']['x']), array_hash(again['train']['x']))

    def test_array_hash_has_shape_and_dtype(self):
        self.assertNotEqual(array_hash(np.ones((2, 2))), array_hash(np.ones((1, 4))))
        self.assertNotEqual(array_hash(np.ones(3)), array_hash(np.ones(3, dtype=np.float32)))

    def test_seal_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); write_json(root/'p.json', {'a':1})
            seal(root, ['p.json']); verify_seal(root)
            write_json(root/'p.json', {'a':2})
            with self.assertRaises(ValueError): verify_seal(root)

    def test_missing_seal_blocks_test(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): verify_seal(Path(d))


class MathContracts(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-1.0,2.0],[-.7,2.1],[-.4,2.2],[-.1,2.3],
                           [.1,3.1],[.4,3.2],[.7,3.3],[1.0,3.4]])
        self.y = np.repeat([0,1],4)

    def test_mode_transform_roundtrip(self):
        t=ModeTransformer(max_modes=2, seed=7).fit(self.x,self.y)
        z=t.transform(self.x,self.y,seed=19)
        x,y=t.inverse(z)
        np.testing.assert_allclose(x,self.x,atol=1e-5)
        np.testing.assert_array_equal(y,self.y)

    def test_mode_transform_column_order_and_constant(self):
        x=np.c_[self.x,np.ones(len(self.x))*4.25]
        t=ModeTransformer(max_modes=2,seed=8).fit(x,self.y)
        x2,y2=t.inverse(t.transform(x,self.y,seed=4))
        np.testing.assert_allclose(x2,x,atol=1e-4)
        np.testing.assert_array_equal(y2,self.y)

    def test_packed_critic_rejects_remainder(self):
        c=PackedCritic(4,width=16,pac=10)
        with self.assertRaises(ValueError): c(torch.ones(11,4))
        self.assertEqual(tuple(c(torch.ones(20,4)).shape),(2,1))

    def test_diffusion_posterior_at_zero_is_x0_without_noise(self):
        schedule=diffusion_schedule(20)
        x0=torch.tensor([[1.0,-2.0]])
        xt=torch.tensor([[9.0,4.0]])
        mean,var=posterior_from_x0(x0,xt,0,schedule)
        torch.testing.assert_close(mean,x0,atol=1e-5,rtol=1e-5)
        self.assertEqual(float(var),0.0)

    def test_diffusion_posterior_matches_hand_formula(self):
        s=diffusion_schedule(20); t=7
        x0=torch.tensor([[.3]]); xt=torch.tensor([[.7]])
        mean,var=posterior_from_x0(x0,xt,t,s)
        b=s['beta'][t]; a=s['alpha'][t]; ab=s['abar'][t]; prev=s['abar'][t-1]
        expected=b*torch.sqrt(prev)/(1-ab)*x0+(1-prev)*torch.sqrt(a)/(1-ab)*xt
        torch.testing.assert_close(mean,expected)
        torch.testing.assert_close(var,b*(1-prev)/(1-ab))

    def test_diffusion_terminal_distribution_is_close_to_noise(self):
        s=diffusion_schedule(100)
        self.assertLess(float(s['abar'][-1]),.001)
        self.assertTrue(torch.all(s['beta']>0)); self.assertTrue(torch.all(s['beta']<1))

    def test_utility_weights_are_normalized_and_prior_shrunk(self):
        w=utility_weights(np.array([[.1,2.0],[2.0,.1]]),np.array([0,20]))
        np.testing.assert_allclose(w.sum(axis=0),1)
        np.testing.assert_allclose(w[:,0],[.5,.5])
        self.assertGreater(w[1,1],w[0,1]);self.assertTrue(np.all(w>0))

    def test_support_gate_rejects_copy_and_extreme_outlier(self):
        x=np.arange(20,dtype=float).reshape(10,2)
        m,_=support_mask(x,np.array([x[0], [10000,10000], [6.2,7.2]]))
        self.assertFalse(m[0]);self.assertFalse(m[1])
        self.assertTrue(m[2])

    def test_portfolio_exact_class_counts_and_source_identity(self):
        pools={'a':(self.x,self.y),'b':(self.x+.05,self.y)}
        x,y,meta=select_portfolio(pools,self.x,self.y,np.array([[.1,.3],[.2,.1]]),
                                np.array([5,5]),np.array([0,0,1,1]),seed=4,weighted=True,gated=False)
        np.testing.assert_array_equal(y,[0,0,1,1]); self.assertEqual(x.shape,(4,2))
        self.assertEqual(len(meta['selected_sources']),4)
        self.assertEqual(len(set(meta['selected_pool_indices'])),4)


class ActualTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng=np.random.default_rng(4)
        cls.x=np.r_[rng.normal(-1,.2,(24,3)),rng.normal(1,.2,(24,3))]
        cls.y=np.repeat([0,1],24)

    def test_all_neural_generators_change_weights_and_sample(self):
        config=GeneratorConfig(steps=30,batch_size=40,width=32,latent=16,diffusion_steps=20,max_modes=2)
        for name,constructor in [('ctgan',CTGAN),('tvae',TVAE),('ddpm',TabDDPM)]:
            with self.subTest(name=name):
                g=constructor(config,seed=2).fit(self.x,self.y)
                self.assertNotEqual(g.audit['initial_sha256'],g.audit['final_sha256'])
                self.assertEqual(g.audit['completed_steps'],30)
                self.assertTrue(np.isfinite(g.audit['losses']).all())
                # Native sampling is checked; strict conditioning is separately tested after adequate training.
                x,y=g.sample_native(12,seed=3)
                self.assertEqual(x.shape,(12,3)); self.assertTrue(np.isfinite(x).all())
                self.assertTrue(set(y).issubset({0,1}))
                x2,y2=g.sample_native(12,seed=3)
                np.testing.assert_array_equal(x,x2);np.testing.assert_array_equal(y,y2)

    def test_baselines_are_real_fitted_samplers(self):
        for constructor in [Copula,Interpolation]:
            g=constructor(seed=3).fit(self.x,self.y)
            labels=np.array([0,1,1,0])
            x,y=g.sample_labels(labels,seed=19)
            np.testing.assert_array_equal(y,labels);self.assertTrue(np.isfinite(x).all())
            with self.assertRaises(ValueError): g.sample_labels(np.array([3]),seed=1)

    def test_student_equal_initialization_and_checkpoint_roundtrip(self):
        config=StudentConfig(steps=25,batch_size=16,width=16,seed=7)
        a=train_student(self.x,self.y,config)
        b=train_student(self.x*1.1,self.y,config)
        self.assertEqual(a.audit['initial_sha256'],b.audit['initial_sha256'])
        self.assertNotEqual(a.audit['initial_sha256'],a.audit['final_sha256'])
        self.assertEqual(a.audit['examples_seen'],400)
        with tempfile.TemporaryDirectory() as d:
            a.save(Path(d)); restored=load_student(Path(d))
            np.testing.assert_array_equal(a.predict_proba(self.x),restored.predict_proba(self.x))

    def test_sampling_budget_never_fabricates_requested_class(self):
        g=TVAE(GeneratorConfig(steps=1,batch_size=40,width=16,latent=8,max_modes=1),seed=1).fit(self.x,self.y)
        g.sample_native=lambda n,seed: (np.zeros((n,3)),np.zeros(n,dtype=int))
        with self.assertRaises(RuntimeError): g.sample_labels(np.array([1,1]),seed=3,max_batches=2)


if __name__=='__main__': unittest.main()
