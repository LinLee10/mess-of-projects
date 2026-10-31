"""Independent H004A artifact verifier. No production imports; all rows checked.

Floating comparisons use rtol=1e-9, atol=1e-10; identities and marginals
are exact. Undefined relative context metrics (zero target) are JSON null.
mechanism_screen contains per-seed reductions and flags plus all-seed flags.
Artifact basenames are relative to AUDIT. No archive provenance is inferred.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
from sklearn.neighbors import KDTree

CAL_SEEDS = [41017, 41029]
AUDIT_SEEDS = [43017, 43029, 43041]
LEVELS = [.05, .25, .5, .75, .95]
METHODS = ['unit_beta', 'heuristic', 'empirical_median', 'empirical_quantile']


class ValidationError(ValueError):
    """Artifact violates the frozen measurement contract."""


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def need(mapping, keys, name):
    require(isinstance(mapping, dict), f'{name}: object required')
    missing = set(keys) - mapping.keys()
    require(not missing, f'{name}: missing fields {sorted(missing)}')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'{path}: duplicate JSON key {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValidationError(f'{path}: nonfinite JSON {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=invalid)


def array(value, shape, name, integer=False, nonnegative=False):
    a = np.asarray(value)
    require(a.shape == shape, f'{name}: shape {a.shape}, expected {shape}')
    require(a.dtype.kind in ('iu' if integer else 'iuf'), f'{name}: real numeric dtype required')
    require(np.isfinite(a).all(), f'{name}: nonfinite')
    if nonnegative:
        require((a >= 0).all(), f'{name}: negative')
    return a


def same(actual, expected, name, exact=False):
    if expected is None:
        require(actual is None, f'{name}: expected null')
        return
    if isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected, f'{name}: identity mismatch')
        return
    a, e = np.asarray(actual), np.asarray(expected)
    array(a, e.shape, name)
    require(np.array_equal(a, e) if exact else np.allclose(a, e, rtol=1e-9, atol=1e-10),
            f'{name}: mismatch')


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def widths(n, h):
    require(np.isfinite(h) and 1 <= h <= n, 'invalid heuristic')
    return np.unique(np.clip([1, h / 2, h, 2*h, 4*h, 8*h], 1, n))


def choose(w, observed, target, objective):
    array(w, (len(w),), 'widths')
    require(len(w) > 0 and (w >= 1).all() and (np.diff(w) > 0).all(), 'invalid grid')
    array(observed, (2, len(w), 5), 'observed quantiles', nonnegative=True)
    array(target, (5,), 'target quantiles', nonnegative=True)
    require((np.diff(observed, axis=-1) >= 0).all() and (np.diff(target) >= 0).all(), 'unsorted quantiles')
    require(objective in ('median', 'quantile'), 'invalid objective')
    if target[2] == 0:
        return float(w[0])
    error = np.abs(observed - target) / target[2]
    loss = error[:, :, 2].mean(axis=0) if objective == 'median' else error.mean(axis=(0, 2))
    return float(w[np.argmin(loss)])


def rank_keys(ids, column):
    # Independent transcription of the frozen H002 identity-keyed tie rule.
    x = ids.astype(np.uint64) ^ np.uint64((column + 1)*7919) ^ np.uint64(217)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


def draw(x, ids, width, seed, context_index=None):
    n = len(x)
    ranks = np.empty(x.shape, dtype=np.int64)
    keys = [rank_keys(ids, j) for j in range(10)]
    for j in range(10):
        ranks[np.lexsort((keys[j], x[:, j])), j] = np.arange(1, n+1)
    rng = np.random.default_rng(seed if context_index is None else np.random.SeedSequence([seed, context_index]))
    latent = rng.beta(ranks / width, (n + 1 - ranks) / width)
    reference = np.sort(x, axis=0)
    assigned = np.empty_like(x)
    inverse = np.empty_like(x)
    for j in range(10):
        assigned[np.lexsort((keys[j], latent[:, j])), j] = reference[:, j]
        inverse[:, j] = reference[np.minimum((n * latent[:, j]).astype(np.int64), n-1), j]
    return assigned, inverse


def nearest(tree, x, scale):
    return tree.query(x / scale, k=1, return_distance=True)[0][:, 0]


def average_ranks(x):
    order = np.argsort(x, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(x[order])) + 1]
    ends = np.r_[starts[1:], len(x)]
    ranks = np.empty(len(x), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
    return ranks


def spearman(x, y):
    columns = np.flatnonzero(np.ptp(x, axis=0) > 0)
    if len(columns) < 2:
        return 0.0
    a = np.column_stack([average_ranks(x[:, j]) for j in columns])
    b = np.column_stack([average_ranks(y[:, j]) for j in columns])
    upper = np.triu_indices(len(columns), 1)
    return float(np.abs(np.corrcoef(a, rowvar=False)[upper] - np.corrcoef(b, rowvar=False)[upper]).mean())


def metrics(g, context, x, table, distance, target_distance, width, grid):
    n = len(x)
    target = np.quantile(target_distance, LEVELS)
    q = np.quantile(distance, LEVELS)
    positive = target[2] > 0
    union = np.union1d(distance, target_distance)
    ks = np.max(np.abs(np.searchsorted(np.sort(distance), union, side='right') / n -
                       np.searchsorted(np.sort(target_distance), union, side='right') / n))
    return dict(context_index=g, **{'class': int(context[2])}, n=n, width=float(width),
                primary_population=bool(n >= 100 and positive), target_quantiles=target.tolist(),
                quantiles=q.tolist(), median_error=float(abs(q[2]-target[2])/target[2]) if positive else None,
                quantile_error=float(np.abs(q-target).mean()/target[2]) if positive else None,
                median_ratio=float(q[2]/target[2]) if positive else None, ks_distance=float(ks),
                below_q05_count=int((distance < target[0]).sum()),
                below_median_count=int((distance < target[2]).sum()),
                zero_distance_count=int((distance == 0).sum()),
                unique_rows=int(len(np.unique(table, axis=0))), spearman_error=spearman(x, table),
                bandwidth_on_upper_grid=bool(width == grid[-1]))


def aggregate(method, seed, records, rows, unique_rows, artifact, artifact_hash):
    primary = [r for r in records if r['primary_population']]
    total = sum(r['n'] for r in primary)
    def weighted(key):
        return sum(r['n']*r[key] for r in primary)/total if total else None
    zero = sum(r['zero_distance_count'] for r in records)
    return dict(method=method, seed=seed, training_rows=rows, primary_rows=total,
                primary_contexts=len(primary), primary_coverage=total/rows,
                weighted_median_error=weighted('median_error'), weighted_quantile_error=weighted('quantile_error'),
                weighted_median_ratio=weighted('median_ratio'), weighted_ks=weighted('ks_distance'),
                weighted_spearman_error=weighted('spearman_error'), all_zero_distance_count=zero,
                all_zero_distance_fraction=zero/rows,
                primary_below_q05_fraction=sum(r['below_q05_count'] for r in primary)/total if total else None,
                primary_below_median_fraction=sum(r['below_median_count'] for r in primary)/total if total else None,
                all_unique_rows=unique_rows, artifact=artifact, artifact_sha256=artifact_hash)


def check_fields(actual, expected, location):
    require(isinstance(actual, dict), f'{location}: object required')
    for key, value in expected.items():
        require(key in actual, f'{location}: missing {key}')
        same(actual[key], value, f'{location}.{key}', exact=isinstance(value, int))


def identities(records, methods, seeds, location):
    require(isinstance(records, list), f'{location}: list required')
    expected = {(m, s) for m in methods for s in seeds}
    indexed = {}
    for record in records:
        need(record, ('method', 'seed'), location)
        method, seed = record['method'], record['seed']
        require(type(method) is str and type(seed) is int, f'{location}: invalid identity')
        identity = (method, seed)
        require(identity in expected, f'{location}: unexpected identity {identity}')
        require(identity not in indexed, f'{location}: duplicate identity {identity}')
        indexed[identity] = record
    require(set(indexed) == expected, f'{location}: missing identities')
    return indexed


def screen(results):
    indexed = identities(results, METHODS, AUDIT_SEEDS, 'screen results')
    answer = {}
    for method in METHODS[2:]:
        records = []
        for seed in AUDIT_SEEDS:
            r, h = indexed[method, seed], indexed['heuristic', seed]
            candidate, baseline = r['weighted_median_error'], h['weighted_median_error']
            reduction = 1 - candidate / baseline if candidate is not None and baseline is not None and baseline > 0 else None
            passed = (reduction is not None and reduction >= .30 and
                      r['all_zero_distance_fraction'] - h['all_zero_distance_fraction'] <= .001)
            records.append(dict(seed=seed, median_error_reduction=reduction, screen_pass=bool(passed)))
        answer[method] = dict(seeds=records, all_seed_screen_pass=all(r['screen_pass'] for r in records))
    return answer


def check_screen(actual, expected):
    need(actual, METHODS[2:], 'mechanism screen')
    require(set(actual) == set(expected), 'mechanism screen: unexpected methods')
    for method, target in expected.items():
        location = f'mechanism screen.{method}'
        saved = actual[method]
        need(saved, ('seeds', 'all_seed_screen_pass'), location)
        require(isinstance(saved['seeds'], list), f'{location}: seeds list required')
        indexed = {}
        for record in saved['seeds']:
            need(record, ('seed', 'median_error_reduction', 'screen_pass'), location)
            seed = record['seed']
            require(type(seed) is int and seed in AUDIT_SEEDS, f'{location}: unexpected seed')
            require(seed not in indexed, f'{location}: duplicate seed')
            indexed[seed] = record
        require(set(indexed) == set(AUDIT_SEEDS), f'{location}: missing seeds')
        for record in target['seeds']:
            check_fields(indexed[record['seed']], record, location)
        same(saved['all_seed_screen_pass'], target['all_seed_screen_pass'], location+'.all_seed_screen_pass')


def verify(inputs, calibration, audit, fixture=False):
    inputs, calibration, audit = map(Path, (inputs, calibration, audit))
    paths = [inputs/'h001-seed17/run/data/train.npy', inputs/'b002-seed17/run/generator.npz',
             inputs/'h003-seed17/run/geometry.npz']
    train = np.load(paths[0], allow_pickle=False)
    require(train.ndim == 2 and train.shape[1] == 13 and len(train) > 0, 'training shape')
    N = len(train)
    array(train, (N, 13), 'train', integer=True)
    model, geo = npz(paths[1]), npz(paths[2])
    need(model, ('contexts', 'counts', 'offsets', 'group_indices', 'sorted'), 'model')
    need(geo, ('scale', 'loo_distance', 'bandwidth', 'pilot_median', 'target_median'), 'geometry')
    contexts = model['contexts']
    require(contexts.ndim == 2 and contexts.shape[1] == 3 and len(contexts) > 0, 'context shape')
    G = len(contexts)
    array(contexts, (G, 3), 'contexts', integer=True)
    require(fixture or (N == 348563 and G == 232), 'default population must be 348563 rows and 232 contexts')
    same(contexts, np.unique(train[:, 10:], axis=0), 'input context identity', exact=True)
    counts = array(model['counts'], (G,), 'counts', integer=True)
    require((counts > 0).all(), 'empty context')
    offsets = array(model['offsets'], (G+1,), 'offsets', integer=True)
    same(offsets, np.r_[0, np.cumsum(counts)], 'offsets', exact=True)
    require(offsets[-1] == N, 'count total')
    ids_all = array(model['group_indices'], (N,), 'group_indices', integer=True)
    same(np.sort(ids_all), np.arange(N), 'identity permutation', exact=True)
    array(model['sorted'], (N, 10), 'sorted', integer=True)
    scale = array(geo['scale'], (10,), 'scale')
    require((scale > 0).all(), 'scale must be positive')
    same(scale, np.maximum(train[:, :10].std(axis=0), 1e-8), 'frozen scale')
    array(geo['loo_distance'], (N,), 'loo_distance', nonnegative=True)
    for key in ('bandwidth', 'pilot_median', 'target_median'):
        array(geo[key], (G,), key, nonnegative=True)
    selection_path = calibration/'SELECTION.json'
    selection_hash = digest(selection_path)
    sel = read_json(selection_path)
    check_fields(sel, dict(calibration_seeds=CAL_SEEDS, audit_seeds=AUDIT_SEEDS,
                          quantile_levels=LEVELS, training_rows=N), 'selection')
    same(read_json(calibration/'COMPLETE.json')['selection_sha256'], selection_hash, 'calibration completion')
    need(sel, ('records',), 'selection')
    require(len(sel['records']) == G, 'selection context coverage')
    groups, checked = [], dict(training_rows=N, contexts=G, loo_rows=0, calibration_tables=0,
                               calibration_rows=0, inverse_rows=0, audit_files=0, audit_rows=0, audit_contexts=0)
    hashes = {str(p): digest(p) for p in paths}
    for g in range(G):
        sl = slice(int(offsets[g]), int(offsets[g+1]))
        ids = ids_all[sl]
        expected_ids = np.flatnonzero(np.all(train[:, 10:] == contexts[g], axis=1))
        same(ids, expected_ids, f'context {g} original ids order', exact=True)
        x, n = train[ids, :10], int(counts[g])
        same(model['sorted'][sl], np.sort(x, axis=0), f'context {g} sorted marginals', exact=True)
        tree = KDTree(x / scale)
        loo = tree.query(x / scale, k=2)[0][:, 1] if n > 1 else np.zeros(1)
        same(geo['loo_distance'][ids], loo, f'context {g} leave one out')
        target = np.quantile(loo, LEVELS)
        same(geo['target_median'][g], target[2], 'geometry target median')
        pilot = 0.0 if n == 1 else float(np.median(nearest(tree, draw(x, ids, 1., 24017+g)[0], scale)))
        same(geo['pilot_median'][g], pilot, 'geometry pilot median')
        h = float(np.clip((target[2]/max(pilot, 1e-12))**2, 1, n)) if target[2] > 0 else 1.
        same(geo['bandwidth'][g], h, 'geometry heuristic')
        grid = widths(n, h)
        k = len(grid)
        path = calibration/f'context_{g:03d}.npz'
        z = npz(path)
        hashes[str(path)] = digest(path)
        record = sel['records'][g]
        need(record, ('empirical_median', 'empirical_quantile'), f'selection {g}')
        check_fields(record, dict(context_index=g, context=contexts[g].tolist(), n=n,
                                 heuristic=h, widths=grid.tolist(), target_quantiles=target.tolist(),
                                 primary_population=bool(n >= 100 and target[2] > 0),
                                 zero_target=bool(target[2] == 0), sha256=hashes[str(path)]), f'selection {g}')
        for key, shape in dict(ids=(n,), widths=(k,), target_distances=(n,), target_quantiles=(5,),
                               seeds=(2,), context=(3,), heuristic=(), pilot_median=(),
                               tables=(2,k,n,10), distances=(2,k,n), inverse_distances=(2,k,n),
                               quantiles=(2,k,5), inverse_quantiles=(2,k,5),
                               anchor_quantiles=(2,k,5), inverse_anchor_quantiles=(2,k,5)).items():
            need(z, (key,), f'calibration {g}')
            array(z[key], shape, f'{g}.{key}', integer=key in ('ids','seeds','context','tables'))
        for key, expected in dict(ids=ids, widths=grid, target_distances=loo, target_quantiles=target,
                                  seeds=CAL_SEEDS, context=contexts[g], heuristic=h, pilot_median=pilot).items():
            same(z[key], expected, f'{g}.{key}', exact=key in ('ids','seeds','context'))
        for s, seed in enumerate(CAL_SEEDS):
            for j, width in enumerate(grid):
                table = z['tables'][s,j]
                same(np.sort(table, axis=0), np.sort(x, axis=0), 'calibration marginal multisets', exact=True)
                assigned, inverse = draw(x, ids, width, seed, g)
                same(table, assigned, 'calibration rank rule and row identity', exact=True)
                distance, inv = nearest(tree, table, scale), nearest(tree, inverse, scale)
                same(z['distances'][s,j], distance, 'calibration distances')
                same(z['inverse_distances'][s,j], inv, 'inverse distances')
                for key, values in [('quantiles', distance), ('inverse_quantiles', inv),
                                    ('anchor_quantiles', np.linalg.norm((table-x)/scale, axis=1)),
                                    ('inverse_anchor_quantiles', np.linalg.norm((inverse-x)/scale, axis=1))]:
                    same(z[key][s,j], np.quantile(values, LEVELS), key)
                checked['calibration_tables'] += 1
                checked['calibration_rows'] += n
                checked['inverse_rows'] += n
        e = choose(grid, z['quantiles'], target, 'median')
        q = choose(grid, z['quantiles'], target, 'quantile')
        same(record['empirical_median'], e, 'selected E width')
        same(record['empirical_quantile'], q, 'selected Q width')
        checked['loo_rows'] += n
        groups.append((ids, x, tree, loo, grid, [1., h, e, q]))
        print(f'calibration context={g} rows={n} widths={k} verified', file=sys.stderr, flush=True)
    results = []
    for mi, method in enumerate(METHODS):
        for seed in AUDIT_SEEDS:
            stem = f'{method}_{seed}'
            path = audit/f'{stem}.npz'
            z, saved = npz(path), read_json(audit/f'{stem}.json')
            need(z, ('table', 'distances', 'counts', 'contexts'), stem)
            need(saved, ('contexts',), stem)
            table = array(z['table'], (N,13), stem+'.table', integer=True)
            array(z['distances'], (N,), stem+'.distances', nonnegative=True)
            same(z['counts'], counts, stem+'.counts', exact=True)
            same(z['contexts'], contexts, stem+'.contexts', exact=True)
            require(len(saved['contexts']) == G, stem+': missing contexts')
            records = []
            for g, (ids, x, tree, loo, grid, selected) in enumerate(groups):
                sl = slice(int(offsets[g]), int(offsets[g+1]))
                block = table[sl, :10]
                same(table[sl,10:], np.tile(contexts[g], (len(ids),1)), 'audit context labels', exact=True)
                same(np.sort(block, axis=0), np.sort(x, axis=0), 'audit marginal multisets', exact=True)
                same(block, draw(x, ids, selected[mi], seed, g)[0], 'audit rank rule and row identity', exact=True)
                distance = nearest(tree, block, scale)
                same(z['distances'][sl], distance, 'audit distances')
                r = metrics(g, contexts[g], x, block, distance, loo, selected[mi], grid)
                check_fields(saved['contexts'][g], r, stem+f'.context {g}')
                records.append(r)
                checked['audit_contexts'] += 1
            artifact_hash = digest(path)
            hashes[str(path)] = artifact_hash
            result = aggregate(method, seed, records, N, int(len(np.unique(table, axis=0))), path.name, artifact_hash)
            check_fields(saved, result, stem)
            results.append(result)
            checked['audit_files'] += 1
            checked['audit_rows'] += N
            print(f'audit {stem} all {N} rows verified', file=sys.stderr, flush=True)
    summary_path = audit/'SUMMARY.json'
    summary = read_json(summary_path)
    same(read_json(audit/'COMPLETE.json')['summary_sha256'], digest(summary_path), 'audit completion')
    need(summary, ('selection_sha256', 'selection_unchanged', 'results', 'mechanism_screen'), 'summary')
    same(summary['selection_sha256'], selection_hash, 'summary selection hash')
    same(summary['selection_unchanged'], True, 'summary selection unchanged')
    require(digest(selection_path) == selection_hash, 'selection changed during verification')
    indexed = identities(summary['results'], METHODS, AUDIT_SEEDS, 'summary results')
    for expected in results:
        actual = indexed[expected['method'], expected['seed']]
        require('contexts' not in actual, 'summary must omit context lists')
        check_fields(actual, expected, 'summary result')
    mechanism = screen(results)
    check_screen(summary['mechanism_screen'], mechanism)
    for folder in (calibration, audit):
        for path in sorted(folder.glob('*.json')):
            hashes[str(path)] = digest(path)
    return dict(status='verified', fixture=fixture, counts=checked, mechanism_screen=mechanism,
                inverse_verification='regenerated frozen H002 identity rank rule; all rows',
                distance_engine='sklearn.neighbors.KDTree exact Euclidean, all rows',
                tolerance=dict(rtol=1e-9, atol=1e-10), artifact_sha256=hashes,
                numpy=np.__version__, sklearn=sklearn.__version__,
                limitations=['No production implementation or real measurements evaluated by authoring tests.',
                             'Archive hashes and internal manifests not verified by this extracted-array verifier.',
                             'Hash consistency cannot establish selection creation time before audit.',
                             'No scientific novelty, privacy, utility or promotion acceptance.'])


def write_new(path, report):
    # Serialize fully before exclusive creation, so invalid JSON leaves no output.
    content = json.dumps(report, allow_nan=False, indent=2) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    created = False
    try:
        with temporary.open('x') as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive publication; never replace.
    finally:
        if created and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'calibration', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    report = verify(args.inputs, args.calibration, args.audit, args.fixture)
    report['elapsed_seconds'] = time.monotonic() - start
    report['verifier_sha256'] = digest(Path(__file__))
    report['contracts_sha256'] = {name: digest(Path(__file__).parent.parent/name)
                                 for name in ('PROTOCOL.md', 'API_CONTRACT.md')}
    write_new(args.output, report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
