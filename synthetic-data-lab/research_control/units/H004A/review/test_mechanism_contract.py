"""Preimplementation H004A acceptance tests; synthetic fixtures only.

Expected distances and selection losses never call production distance/selection
helpers. H002 ordinal and assign_columns are explicitly frozen dependencies.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

UNIT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Deliberately fail collection if the contracted implementation does not exist.
m = load_module("h004a_mechanism_contract", UNIT / "src" / "mechanism.py")
h002 = load_module("h002_frozen_contract", UNIT.parent / "H002" / "src" / "rank_synthesis.py")


def direct_distances(reference, query, scale, exclude_identity=False):
    """Small scalar-loop oracle, independent of any nearest-neighbor library."""
    answers = []
    for i, row in enumerate(query):
        distances = []
        for j, other in enumerate(reference):
            if exclude_identity and i == j:
                continue
            distances.append(sum(((float(a) - float(b)) / float(s)) ** 2
                                 for a, b, s in zip(row, other, scale)) ** 0.5)
        answers.append(min(distances) if distances else 0.0)
    return np.asarray(answers)


def curve():
    target = np.array([1., 2., 4., 8., 16.])
    return np.array([1., 2., 4.]), np.tile(target, (2, 3, 1)), target


def sample_inputs():
    return np.array([[4, 8], [1, 8], [4, -2], [9, 3]], dtype=np.int64), np.array([31, 7, 24, 2])


def assert_value_errors(call, cases):
    for kwargs in cases:
        with pytest.raises(ValueError):
            call(**kwargs)


@pytest.mark.parametrize("n,h,expected", [
    (1, 1, [1.]), (10, 3, [1., 1.5, 3., 6., 10.]),
    (16, 4, [1., 2., 4., 8., 16.]), (5, 5, [1., 2.5, 5.]),
])
def test_sorted_unique_clipped_grid(n, h, expected):
    result = m.candidate_widths(n, h)
    assert isinstance(result, np.ndarray) and result.ndim == 1
    assert result.dtype.kind == 'f'
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("name", ["candidate_widths", "beta_variance_ratio"])
def test_scalar_validation(name):
    fn = getattr(m, name)
    for n in (True, 0, -1, 2.5, [3], np.array([3]), complex(3), np.nan, np.inf):
        with pytest.raises(ValueError):
            fn(n, 1)
    for width in (True, 0, -1, 4, [1], np.array([1]), complex(1), np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            fn(3, width)


def test_numpy_real_integral_scalars():
    np.testing.assert_array_equal(m.candidate_widths(np.int64(4), np.float64(2)), [1., 2., 4.])
    assert m.beta_variance_ratio(np.int64(4), np.float64(2)) == pytest.approx(12 / 7)


def test_variance_ratio_exact_formula():
    for n, width in ((1, 1), (7, 1), (7, 2.5), (7, 7)):
        assert m.beta_variance_ratio(n, width) == pytest.approx(width * (n + 2) / (n + 1 + width))


@pytest.mark.parametrize("objective", ["median", "quantile"])
def test_objective_arithmetic_across_seeds(objective):
    widths = np.array([1., 2., 3.])
    target = np.array([1., 2., 4., 8., 16.])
    # Median prefers width 1; five-quantile loss prefers width 2.
    observed = np.array([
        [[0, 0, 4, 20, 30], [1, 2, 5, 8, 16], [1, 2, 7, 8, 16]],
        [[0, 0, 4, 20, 30], [1, 2, 6, 8, 16], [1, 2, 3, 8, 16]],
    ], dtype=float)
    losses = []
    for k in range(3):
        indices = [2] if objective == 'median' else range(5)
        terms = [abs(observed[s, k, q] - target[q]) / target[2]
                 for s in range(2) for q in indices]
        losses.append(sum(terms) / len(terms))
    assert m.choose_width(widths, observed, target, objective) == widths[min(range(3), key=losses.__getitem__)]


def test_default_objective_is_median():
    w, obs, target = curve()
    obs[:, 0, 2] = 3
    assert m.choose_width(w, obs, target) == m.choose_width(w, obs, target, 'median') == 2


@pytest.mark.parametrize("objective", ["median", "quantile"])
def test_nonmonotonic_width_response(objective):
    w = np.array([1., 2., 3., 4., 8.])
    target = np.array([1., 2., 3., 4., 5.])
    obs = np.array([[target + delta for delta in (5, 1, 4, 0, 3)]])
    assert m.choose_width(w, obs, target, objective) == 4


@pytest.mark.parametrize("objective", ["median", "quantile"])
def test_exact_tie_smallest_width(objective):
    w, obs, target = curve()
    obs[:, 0] = target + 2
    obs[:, 1] = target + 1
    obs[:, 2] = target + 1
    assert m.choose_width(w, obs, target, objective) == 2


def test_width_vector_validation():
    w, obs, target = curve()
    for bad in ([], [1, 1, 2], [2, 1, 3], [0, 1, 2], [1, np.nan, 3],
                [1, np.inf, 3], [1, 2, 3j], [[1, 2, 3]], 1):
        with pytest.raises(ValueError):
            m.choose_width(bad, obs, target)


def test_exact_shapes_no_broadcasting():
    w, obs, target = curve()
    for bad in (obs[0], obs[:, :1], obs[:, :, :1], obs[:, :, :4], np.empty((0, 3, 5)), obs[..., None]):
        with pytest.raises(ValueError):
            m.choose_width(w, bad, target)
    for bad in (4., target[None, :], target[:, None], target[:1], target[:4]):
        with pytest.raises(ValueError):
            m.choose_width(w, obs, bad)


@pytest.mark.parametrize("location", ["observed", "target"])
def test_distance_curve_values_validation(location):
    for value in (np.nan, np.inf, -np.inf, -1., 1j):
        w, obs, target = curve()
        if location == 'observed':
            obs = obs.astype(complex) if value == 1j else obs
            obs[0, 0, 0] = value
        else:
            target = target.astype(complex) if value == 1j else target
            target[0] = value
        with pytest.raises(ValueError):
            m.choose_width(w, obs, target)


def test_quantile_order_validation():
    w, obs, target = curve()
    obs[0, 0, :2] = [2, 1]
    with pytest.raises(ValueError):
        m.choose_width(w, obs, target)
    _, obs, target = curve()
    target[:2] = [2, 1]
    with pytest.raises(ValueError):
        m.choose_width(w, obs, target)


def test_zero_target_valid_fallback():
    assert m.choose_width([2., 4.], np.zeros((1, 2, 5)), np.zeros(5)) == 2


def test_validate_before_zero_target_fallback():
    w, obs, _ = curve()
    target = np.zeros(5)
    for bad in (np.full_like(obs, np.nan), -np.ones_like(obs), obs[0], obs.astype(complex)):
        with pytest.raises(ValueError):
            m.choose_width(w, bad, target)
    for bad_w in ([2, 1, 3], [1, 1, 3]):
        with pytest.raises(ValueError):
            m.choose_width(bad_w, obs, target)
    with pytest.raises(ValueError):
        m.choose_width(w, obs, target, 'other')
    with pytest.raises(ValueError):
        m.choose_width(w, obs, [0, 0, 0, -1, 0])


def test_invalid_objectives():
    w, obs, target = curve()
    for objective in ('', 'Median', 'other', None, 1):
        with pytest.raises(ValueError):
            m.choose_width(w, obs, target, objective)


def test_generation_matches_frozen_identity_dependencies():
    reference, ids = sample_inputs()
    ranks = np.column_stack([h002.ordinal(reference[:, j], ids, j) for j in range(2)])
    expected = np.random.default_rng(np.random.SeedSequence([41017, 9])).beta(ranks / 2.5, (5 - ranks) / 2.5)
    result = m.sample_context(reference, ids, 2.5, 41017, 9)
    assert {'latent', 'assigned', 'inverse'} <= result.keys()
    np.testing.assert_array_equal(result['latent'], expected)
    np.testing.assert_array_equal(result['assigned'], h002.assign_columns(expected, reference, ids))
    for j in range(2):
        indices = np.minimum(np.floor(4 * expected[:, j]).astype(int), 3)
        np.testing.assert_array_equal(result['inverse'][:, j], np.sort(reference[:, j])[indices])


@pytest.mark.parametrize("reference,ids", [
    ([[5, -2]], [91]), ([[7, 7], [7, 7], [7, 7]], [8, 2, 19]),
    ([[2, 3], [2, 3], [8, 3], [1, 3]], [20, 4, 18, 7]),
])
def test_degenerate_fixtures_preserve_identities_and_marginals(reference, ids):
    reference, ids = np.asarray(reference), np.asarray(ids)
    ranks = np.column_stack([h002.ordinal(reference[:, j], ids, j) for j in range(reference.shape[1])])
    latent = np.random.default_rng(np.random.SeedSequence([43017, 0])).beta(ranks, len(ids) + 1 - ranks)
    result = m.sample_context(reference, ids, 1, 43017)
    np.testing.assert_array_equal(result['latent'], latent)
    np.testing.assert_array_equal(result['assigned'], h002.assign_columns(latent, reference, ids))
    for key in ('latent', 'assigned', 'inverse'):
        assert result[key].shape == reference.shape
    for j in range(reference.shape[1]):
        np.testing.assert_array_equal(np.sort(result['assigned'][:, j]), np.sort(reference[:, j]))
    assert np.isfinite(result['latent']).all()
    assert ((result['latent'] >= 0) & (result['latent'] <= 1)).all()


def test_deterministic_generation_and_context_seed():
    reference, ids = sample_inputs()
    a = m.sample_context(reference, ids, 2, 17, 3)
    b = m.sample_context(reference, ids, 2, 17, 3)
    for key in ('latent', 'assigned', 'inverse'):
        np.testing.assert_array_equal(a[key], b[key])
    for seed, context in ((18, 3), (17, 4)):
        assert not np.array_equal(a['latent'], m.sample_context(reference, ids, 2, seed, context)['latent'])


def test_sample_array_validation():
    reference, ids = sample_inputs()
    base = dict(reference=reference, ids=ids, width=1, seed=17)
    invalid = []
    for bad in (reference.astype(float), reference.astype(complex), reference.astype(bool),
                reference.ravel(), np.empty((0, 2), int), np.empty((4, 0), int),
                np.full((4, 2), 2**63, dtype=np.uint64)):
        invalid.append(dict(base, reference=bad))
    for bad in (ids.astype(float), ids.astype(bool), ids.astype(complex), ids[:, None], ids[:1],
                [1, 1, 2, 3], [-1, 1, 2, 3]):
        invalid.append(dict(base, ids=bad))
    assert_value_errors(m.sample_context, invalid)


def test_sample_scalar_validation():
    reference, ids = sample_inputs()
    base = dict(reference=reference, ids=ids, width=1, seed=17, context_index=0)
    cases = []
    for name in ('seed', 'context_index'):
        for bad in (True, -1, 1.5, np.nan, np.inf, 1j, [1], np.array([1])):
            cases.append(dict(base, **{name: bad}))
    for bad in (True, 0, 5, np.nan, np.inf, 1j, [1], np.array([1])):
        cases.append(dict(base, width=bad))
    assert_value_errors(m.sample_context, cases)


def test_exact_marginal_multisets_and_int64_boundaries():
    edge = np.iinfo(np.int64)
    reference = np.array([[edge.min, 5], [edge.max, 5], [0, 7], [0, 9]], dtype=np.int64)
    result = m.sample_context(reference, np.array([7, 3, 9, 1]), 4, 43029)
    for j in range(2):
        np.testing.assert_array_equal(np.sort(result['assigned'][:, j]), np.sort(reference[:, j]))


def test_nearest_direct_distance_oracle():
    reference = np.array([[0., 0.], [4., 6.], [4., 6.], [-2., 9.]])
    query = np.array([[1., 3.], [4., 6.], [8., -2.], [-1., 8.], [30., 2.]])
    scale = np.array([2., 3.])
    got = m.nearest_distances(reference, query, scale)
    assert np.asarray(got).shape == (len(query),)
    np.testing.assert_allclose(got, direct_distances(reference, query, scale), rtol=1e-13, atol=1e-13)


def test_leave_one_out_identity_exclusion_and_singleton():
    reference = np.array([[0., 0.], [0., 0.], [6., 8.], [12., 16.]])
    scale = np.array([2., 4.])
    got = m.leave_one_out(reference, scale)
    assert np.asarray(got).shape == (4,)
    np.testing.assert_allclose(got, direct_distances(reference, reference, scale, True))
    np.testing.assert_array_equal(np.asarray(got)[:2], [0., 0.])
    assert np.asarray(got)[2] > 0
    np.testing.assert_array_equal(m.leave_one_out(reference[:1], scale), [0.])


def test_scale_invariance():
    reference = np.array([[0., 3.], [2., -1.], [7., 8.]])
    query = np.array([[1., 1.], [9., 2.]])
    scale = np.array([2., 5.])
    factors = np.array([8., .25])
    np.testing.assert_allclose(m.nearest_distances(reference * factors, query * factors, scale * factors),
                               direct_distances(reference, query, scale))
    np.testing.assert_allclose(m.leave_one_out(reference * factors, scale * factors),
                               direct_distances(reference, reference, scale, True))


def test_distance_array_and_scale_validation():
    reference = np.array([[0., 1.], [2., 3.]])
    query = np.array([[1., 2.]])
    scale = np.ones(2)
    for bad in (np.empty((0, 2)), reference.ravel(), reference[..., None],
                reference.astype(complex), np.array([[np.nan, 1]]), np.array([[np.inf, 1]])):
        with pytest.raises(ValueError):
            m.nearest_distances(bad, query, scale)
        with pytest.raises(ValueError):
            m.leave_one_out(bad, scale)
    for bad in (query.ravel(), np.empty((0, 2)), np.ones((1, 1)), query.astype(complex),
                np.array([[1., np.nan]]), np.array([[1., -np.inf]])):
        with pytest.raises(ValueError):
            m.nearest_distances(reference, bad, scale)
    for bad in (1., [1.], [[1., 1.]], [0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], [1j, 1]):
        with pytest.raises(ValueError):
            m.nearest_distances(reference, query, bad)
        with pytest.raises(ValueError):
            m.leave_one_out(reference, bad)


def test_original_inputs_preserved():
    reference, ids = sample_inputs()
    query = np.array([[2., 3.], [8., 5.]])
    scale = np.array([2., 3.])
    w, obs, target = curve()
    originals = [reference, ids, query, scale, w, obs, target]
    copies = [a.copy() for a in originals]
    for a in originals:
        a.flags.writeable = False
    m.sample_context(reference, ids, 2, 17)
    m.nearest_distances(reference, query, scale)
    m.leave_one_out(reference, scale)
    m.choose_width(w, obs, target)
    for a, copy in zip(originals, copies):
        np.testing.assert_array_equal(a, copy)


def test_atomic_json_success(tmp_path):
    path = tmp_path / 'nested' / 'result.json'
    value = {'widths': [1, 2.5], 'complete': True, 'note': 'fixture'}
    m.atomic_json(path, value)
    assert path.read_bytes().endswith(b'\n')
    assert json.loads(path.read_text()) == value
    assert set(path.parent.iterdir()) == {path}


def test_atomic_json_serialization_failure_cleanup(tmp_path):
    path = tmp_path / 'result.json'
    for value in ({'bad': np.nan}, {'bad': np.inf}, {'bad': -np.inf}, {'bad': object()}):
        # The contract declares ValueError except for existing-path rejection.
        with pytest.raises(ValueError):
            m.atomic_json(path, value)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


def test_atomic_json_preserves_existing_file(tmp_path):
    path = tmp_path / 'result.json'
    original = b'original evidence\n'
    path.write_bytes(original)
    for value in ({'replacement': 1}, {'bad': np.nan}):
        with pytest.raises(FileExistsError):
            m.atomic_json(path, value)
        assert path.read_bytes() == original
        assert set(tmp_path.iterdir()) == {path}


def test_generated_assignment_distances_use_independent_oracle():
    reference, ids = sample_inputs()
    result = m.sample_context(reference, ids, 2.5, 43041, 7)
    scale = np.array([2., 3.])
    for key in ('assigned', 'inverse'):
        np.testing.assert_allclose(
            m.nearest_distances(reference, result[key], scale),
            direct_distances(reference, result[key], scale), rtol=1e-13, atol=1e-13)


def test_distance_returns_every_query_and_reference_row():
    # Analytic fixture with a late unique nearest reference and a late query.
    reference = np.arange(10003, dtype=float)[:, None]
    query = np.array([[10002.25], [-2.], [5000.5]])
    np.testing.assert_allclose(m.nearest_distances(reference, query, [1.]), [.25, 2., .5])
    many_queries = np.arange(10003, dtype=float)[:, None] + .25
    got = np.asarray(m.nearest_distances([[0.]], many_queries, [1.]))
    assert got.shape == (10003,)
    np.testing.assert_array_equal(got, many_queries[:, 0])
