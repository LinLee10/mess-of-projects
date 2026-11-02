"""Synthetic acceptance contract only; no research or behavioral-pass claims."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.special import betainc

UNIT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

frozen = load('h004b_frozen_h002', UNIT.parent / 'H002/src/rank_synthesis.py')
m = load('h004b_candidate', UNIT / 'src/resampling.py')
X = (np.arange(70).reshape(7, 10) * 13 % 9).astype(np.int64)
IDS = np.array([101, 4, 98, 13, 700, 2, 41])

def oracle(x=X, ids=IDS, seed=17, g=3):
    n = len(x)
    r = np.column_stack([frozen.ordinal(x[:, j], ids, j) for j in range(10)])
    rng = np.random.default_rng(np.random.SeedSequence([seed, g, 3202]))
    a = rng.integers(0, n, size=n)
    u = rng.beta(r[a], n + 1 - r[a])
    inv = np.sort(x, axis=0)[np.minimum(np.floor(n*u).astype(int), n-1), np.arange(10)]
    return dict(anchor_ids=ids[a], latent=u, inverse=inv,
                assigned=frozen.assign_columns(u, x, np.arange(n)))

def check_exact(x=X, ids=IDS, seed=17, g=3):
    got, expected = m.iid_context(x, ids, seed, g), oracle(x, ids, seed, g)
    for key in expected:
        np.testing.assert_array_equal(got[key], expected[key])
    return got

def dep(x, y, error, collapsed, pairs):
    got = m.dependence_error(np.asarray(x), np.asarray(y))
    assert got['mean_absolute_error'] == pytest.approx(error, abs=1e-14)
    assert got['collapsed_synthetic_columns'] == collapsed
    assert all(type(j) is int for j in got['collapsed_synthetic_columns'])
    assert got['pairs'] == pairs

def test_iid_exact_draw_order_and_real_identity_ties():
    check_exact()

def test_iid_shapes_and_integer_outputs():
    out = check_exact()
    assert out['anchor_ids'].shape == (7,)
    for key in ('latent', 'inverse', 'assigned'):
        assert out[key].shape == X.shape
    for key in ('anchor_ids', 'inverse', 'assigned'):
        assert out[key].dtype.kind in 'iu'

def test_iid_determinism():
    a, b = m.iid_context(X, IDS, 9, 0), m.iid_context(X, IDS, 9, 0)
    for key in ('anchor_ids', 'latent', 'inverse', 'assigned'):
        np.testing.assert_array_equal(a[key], b[key])

def test_iid_seed_and_group_streams():
    for seed, g in ((0, 0), (1, 0), (0, 1), (2**40, 2**35)):
        check_exact(seed=seed, g=g)

def test_iid_is_repeated_anchor_sampling():
    expected = oracle()
    assert len(np.unique(expected['anchor_ids'])) < len(IDS)
    np.testing.assert_array_equal(m.iid_context(X, IDS, 17, 3)['anchor_ids'], expected['anchor_ids'])

def test_iid_singleton():
    check_exact(X[:1], IDS[:1])

def test_iid_constant_and_duplicate_records():
    check_exact(np.full((7, 10), -8), IDS)
    check_exact(np.repeat(X[:2], [3, 4], axis=0), IDS)

def test_iid_pairing_boundaries_and_exact_marginals():
    out = m.iid_context(X, IDS, 17, 3)
    assert np.all((out['latent'] >= 0) & (out['latent'] <= 1))
    for key in ('inverse', 'assigned'):
        assert np.all(out[key] >= X.min(axis=0))
        assert np.all(out[key] <= X.max(axis=0))
    np.testing.assert_array_equal(np.sort(out['assigned'], axis=0), np.sort(X, axis=0))
    np.testing.assert_array_equal(out['assigned'], frozen.assign_columns(out['latent'], X, np.arange(7)))
    idx = np.minimum(np.floor(7*out['latent']).astype(int), 6)
    np.testing.assert_array_equal(out['inverse'], np.sort(X, axis=0)[idx, np.arange(10)])

def test_iid_inverse_endpoint_clipping(monkeypatch):
    u = np.tile([0., 1., .5, .1, .9, 0., 1., .5, .1, .9], (7, 1))
    class Stub:
        def integers(self, low, high, size): return np.arange(size)
        def beta(self, a, b): return u.copy()
    monkeypatch.setattr(m.np.random, 'default_rng', lambda *args: Stub())
    out = m.iid_context(X, IDS, 17, 3)
    idx = np.minimum(np.floor(7*u).astype(int), 6)
    np.testing.assert_array_equal(out['inverse'], np.sort(X, axis=0)[idx, np.arange(10)])

def test_iid_input_preservation():
    x, ids = X.copy(), IDS.copy()
    m.iid_context(x, ids, 17, 3)
    np.testing.assert_array_equal(x, X)
    np.testing.assert_array_equal(ids, IDS)

def test_iid_invalid_arrays():
    for x in (X.astype(float), X.astype(complex), X.astype(bool), X.astype(object),
              [], [[1], [1, 2]], np.empty((0, 10), int), X[:, :9], X.ravel(), X[None]):
        with pytest.raises(ValueError): m.iid_context(x, IDS, 0, 0)

def test_iid_invalid_identities():
    for ids in (IDS.astype(float), IDS.astype(complex), IDS.astype(bool), IDS[:-1],
                IDS[:, None], [-1, 1, 2, 3, 4, 5, 6], [1]*7, [[1], [2, 3]]):
        with pytest.raises(ValueError): m.iid_context(X, ids, 0, 0)

def test_iid_invalid_seed_and_group():
    for bad in (-1, True, 1., 1j, np.int64(1), None, '1'):
        for seed, g in ((bad, 0), (0, bad)):
            with pytest.raises(ValueError): m.iid_context(X, IDS, seed, g)

def test_uniform_beta_mixture_cdf_identity():
    for n in (1, 2, 7, 31):
        u = np.array([0., 1e-8, .01, .19, .5, .83, 1-1e-8, 1.])
        r = np.arange(1, n+1)
        np.testing.assert_allclose(betainc(r[:, None], (n+1-r)[:, None], u).mean(axis=0), u, atol=2e-14, rtol=0)

def test_dependence_perfect_monotonic():
    dep([[1, 2], [2, 4], [3, 8]], [[4, 1], [5, 2], [8, 3]], 0., [], 1)

def test_dependence_reversed_relation():
    dep([[1, 1], [2, 2], [3, 3]], [[1, 3], [2, 2], [3, 1]], 2., [], 1)

def test_dependence_average_ties():
    # Centered ranks: [-1,-1,.5,1.5] and [-1.5,0,0,1.5]; corr = 5/6.
    dep([[0, 0], [0, 1], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2], [3, 3]], 1/6, [], 1)

def test_dependence_collapsed_coordinate_convention():
    dep([[1, 1], [2, 2], [3, 3]], [[1, 0], [2, 0], [3, 0]], 1., [1], 1)

def test_dependence_training_constants_excluded_and_pairs_averaged():
    x = np.column_stack([np.arange(4)]*3 + [np.ones(4)])
    y = np.column_stack([np.arange(4), np.arange(4)[::-1], np.ones(4), np.arange(4)])
    dep(x, y, 4/3, [2], 3)

def test_dependence_fewer_than_two_varying_columns():
    dep([[1, 0], [2, 0]], [[7, 0], [7, 1]], 0., [0], 0)
    dep([[1, 2]], [[8, 9]], 0., [], 0)
    dep([[1], [1]], [[2], [3]], 0., [], 0)

def test_dependence_invalid_shape_and_no_broadcasting():
    for bad in ([], [1, 2], [[1], [2, 3]], np.empty((0, 2)), np.empty((2, 0)),
                np.ones((3, 1)), np.ones((1, 2)), np.ones((2, 2, 1))):
        for x, y in ((bad, np.ones((2, 2))), (np.ones((2, 2)), bad)):
            with pytest.raises(ValueError): m.dependence_error(x, y)

def test_dependence_nonfinite_complex_and_nonnumeric():
    for bad in (np.array([[1, np.nan], [2, 3]]), np.full((2, 2), np.inf),
                np.ones((2, 2), complex), np.full((2, 2), 'a')):
        for x, y in ((bad, X[:2, :2]), (X[:2, :2], bad)):
            with pytest.raises(ValueError): m.dependence_error(x, y)

def test_dependence_input_preservation():
    x, y = X[:, :3].astype(float), X[:, :3].astype(float)[::-1].copy()
    before = (x.copy(), y.copy())
    m.dependence_error(x, y)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(y, before[1])
