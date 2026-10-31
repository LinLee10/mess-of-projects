"""Independent synthetic artifact construction; no production imports or mocks.

Distances use exhaustive pairwise numpy, ranks use scalar uint64 arithmetic,
metrics use explicit empirical CDFs and pairwise average-tie rank formulas.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('artifact_verifier', Path(__file__).with_name('verify_response.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
QS = [.05, .25, .5, .75, .95]
CS = [41017, 41029]
AS = [43017, 43029, 43041]
MS = ['unit_beta', 'heuristic', 'empirical_median', 'empirical_quantile']


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def hash_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(identity, j):
    mask = (1 << 64)-1
    z = (int(identity) ^ ((j+1)*7919) ^ 217) + 0x9e3779b97f4a7c15
    z &= mask
    z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
    z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
    return z ^ (z >> 31)


def generate(x, ids, w, seed, g=None):
    n = len(x)
    ranks = np.zeros_like(x)
    for j in range(10):
        order = sorted(range(n), key=lambda i: (x[i,j], key(ids[i], j)))
        for rank, i in enumerate(order, 1):
            ranks[i,j] = rank
    rng = np.random.default_rng(seed if g is None else np.random.SeedSequence([seed,g]))
    u = rng.beta(ranks/w, (n+1-ranks)/w)
    a, inverse = np.zeros_like(x), np.zeros_like(x)
    for j in range(10):
        values = sorted(x[:,j])
        order = sorted(range(n), key=lambda i: (u[i,j], key(ids[i],j)))
        for rank, i in enumerate(order):
            a[i,j] = values[rank]
        for i in range(n):
            inverse[i,j] = values[min(int(n*u[i,j]),n-1)]
    return a, inverse


def brute(x, y, scale, loo=False):
    d = np.sqrt(np.sum(((y[:,None,:]-x[None,:,:])/scale)**2, axis=2))
    if loo:
        if len(x) == 1:
            return np.zeros(1)
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def correlations(x, y):
    cols = [j for j in range(10) if len(set(x[:,j])) > 1]
    def rank(column):
        return np.array([1 + (column < a).sum() + ((column == a).sum()-1)/2 for a in column])
    errors = []
    for i, j in enumerate(cols):
        for k in cols[i+1:]:
            errors.append(abs(np.corrcoef(rank(x[:,j]),rank(x[:,k]))[0,1] -
                              np.corrcoef(rank(y[:,j]),rank(y[:,k]))[0,1]))
    return float(np.mean(errors)) if errors else 0.


def context_metrics(g, c, x, y, d, target_d, w, grid):
    t, q = np.quantile(target_d, QS), np.quantile(d, QS)
    denominator = t[2]
    ks = max(abs(np.mean(d <= z)-np.mean(target_d <= z)) for z in set(d) | set(target_d))
    return {'context_index':g, 'class':int(c[2]), 'n':len(x), 'width':float(w),
            'primary_population':bool(len(x)>=100 and denominator>0),
            'target_quantiles':t.tolist(), 'quantiles':q.tolist(),
            'median_error':float(abs(q[2]-t[2])/denominator) if denominator else None,
            'quantile_error':float(np.mean(abs(q-t))/denominator) if denominator else None,
            'median_ratio':float(q[2]/denominator) if denominator else None,
            'ks_distance':float(ks), 'below_q05_count':int(sum(d<t[0])),
            'below_median_count':int(sum(d<t[2])), 'zero_distance_count':int(sum(d==0)),
            'unique_rows':len(set(map(tuple,y))), 'spearman_error':correlations(x,y),
            'bandwidth_on_upper_grid':bool(w==max(grid))}


@pytest.fixture
def bundle(tmp_path):
    inp, cal, audit = [tmp_path / s for s in ('inputs','calibration','audit')]
    cal.mkdir(); audit.mkdir()
    rng = np.random.default_rng(8742)
    ordinary = rng.integers(0,100,(100,10))
    ordinary[:,9] = 4  # constant feature with remaining varying correlations
    blocks = [ordinary, np.tile(np.arange(10), (4,1)), np.ones((1,10), dtype=int),
              np.tile(np.arange(10),(5,1))]
    contexts = np.array([[0,0,1],[0,1,2],[1,0,3],[1,1,4]])
    counts = np.array([len(x) for x in blocks])
    offsets = np.r_[0,np.cumsum(counts)]
    # Interleave training so concatenated context table order differs from source.
    raw = np.concatenate([np.column_stack([x,np.tile(c,(len(x),1))]) for x,c in zip(blocks,contexts)])
    train = raw[rng.permutation(len(raw))]
    ids = [np.flatnonzero(np.all(train[:,10:]==c,axis=1)) for c in contexts]
    blocks = [train[ii,:10] for ii in ids]
    paths = [inp/'h001-seed17/run/data/train.npy',inp/'b002-seed17/run/generator.npz',
             inp/'h003-seed17/run/geometry.npz']
    for p in paths:
        p.parent.mkdir(parents=True)
    np.save(paths[0],train)
    np.savez(paths[1],contexts=contexts,counts=counts,offsets=offsets,
             group_indices=np.concatenate(ids),sorted=np.concatenate([np.sort(x,axis=0) for x in blocks]))
    scale = np.maximum(train[:,:10].std(axis=0),1e-8)
    loo_all = np.zeros(len(train)); hs=[]; pilots=[]; targets=[]; records=[]; choices=[]
    for g,(x,ii,c) in enumerate(zip(blocks,ids,contexts)):
        n=len(x)
        loo = brute(x,x,scale,True); loo_all[ii]=loo
        target = np.quantile(loo,QS)
        pilot = float(np.median(brute(x,generate(x,ii,1,24017+g)[0],scale))) if n>1 else 0.
        h = float(np.clip((target[2]/max(pilot,1e-12))**2,1,n)) if target[2]>0 else 1.
        grid = np.array(sorted(set(max(1,min(n,z)) for z in [1,h/2,h,2*h,4*h,8*h])))
        hs.append(h); pilots.append(pilot); targets.append(target[2])
        tables=[]; ds=[]; invs=[]; qs=[]; iqs=[]; anchors=[]; ianchors=[]
        for seed in CS:
            st=[]; sd=[]; si=[]; sq=[]; siq=[]; sa=[]; sia=[]
            for w in grid:
                a,b=generate(x,ii,w,seed,g)
                d,di=brute(x,a,scale),brute(x,b,scale)
                st.append(a); sd.append(d); si.append(di); sq.append(np.quantile(d,QS)); siq.append(np.quantile(di,QS))
                sa.append(np.quantile(np.sqrt(np.sum(((a-x)/scale)**2,axis=1)),QS))
                sia.append(np.quantile(np.sqrt(np.sum(((b-x)/scale)**2,axis=1)),QS))
            tables.append(st); ds.append(sd); invs.append(si); qs.append(sq); iqs.append(siq); anchors.append(sa); ianchors.append(sia)
        observed=np.array(qs)
        if target[2]:
            e=float(grid[min(range(len(grid)),key=lambda j: sum(abs(observed[s,j,2]-target[2]) for s in range(2)))])
            q=float(grid[min(range(len(grid)),key=lambda j: sum(abs(observed[s,j,l]-target[l]) for s in range(2) for l in range(5)))])
        else:
            e=q=float(grid[0])
        choices.append([1.,h,e,q])
        p=cal/f'context_{g:03d}.npz'
        np.savez(p,ids=ii,widths=grid,target_distances=loo,target_quantiles=target,seeds=CS,context=c,
                 heuristic=h,pilot_median=pilot,tables=tables,distances=ds,inverse_distances=invs,
                 quantiles=qs,inverse_quantiles=iqs,anchor_quantiles=anchors,inverse_anchor_quantiles=ianchors)
        records.append(dict(context_index=g,context=c.tolist(),n=n,heuristic=h,widths=grid.tolist(),
                            target_quantiles=target.tolist(),empirical_median=e,empirical_quantile=q,
                            primary_population=bool(n>=100 and target[2]>0),zero_target=bool(target[2]==0),sha256=hash_file(p)))
    np.savez(paths[2],scale=scale,loo_distance=loo_all,bandwidth=hs,pilot_median=pilots,target_median=targets)
    save_json(cal/'SELECTION.json',dict(calibration_seeds=CS,audit_seeds=AS,quantile_levels=QS,training_rows=len(train),records=records))
    selection_hash=hash_file(cal/'SELECTION.json')
    save_json(cal/'COMPLETE.json',dict(selection_sha256=selection_hash))
    results=[]
    for mi,method in enumerate(MS):
        for seed in AS:
            tables=[]; distances=[]; cm=[]
            for g,(x,ii,c) in enumerate(zip(blocks,ids,contexts)):
                y=generate(x,ii,choices[g][mi],seed,g)[0]
                d=brute(x,y,scale)
                tables.append(np.column_stack([y,np.tile(c,(len(x),1))])); distances.append(d)
                cm.append(context_metrics(g,c,x,y,d,loo_all[ii],choices[g][mi],records[g]['widths']))
            table=np.concatenate(tables); distance=np.concatenate(distances)
            stem=f'{method}_{seed}'; p=audit/f'{stem}.npz'
            np.savez(p,table=table,distances=distance,counts=counts,contexts=contexts)
            primary=[r for r in cm if r['primary_population']]; nr=sum(r['n'] for r in primary)
            def weighted(k):
                return sum(r[k]*r['n'] for r in primary)/nr if nr else None
            r=dict(method=method,seed=seed,training_rows=len(train),primary_rows=nr,primary_contexts=len(primary),
                   primary_coverage=nr/len(train),weighted_median_error=weighted('median_error'),
                   weighted_quantile_error=weighted('quantile_error'),weighted_median_ratio=weighted('median_ratio'),
                   weighted_ks=weighted('ks_distance'),weighted_spearman_error=weighted('spearman_error'),
                   all_zero_distance_count=int(sum(distance==0)),all_zero_distance_fraction=float(np.mean(distance==0)),
                   primary_below_q05_fraction=sum(z['below_q05_count'] for z in primary)/nr if nr else None,
                   primary_below_median_fraction=sum(z['below_median_count'] for z in primary)/nr if nr else None,
                   all_unique_rows=len(set(map(tuple,table))),artifact=p.name,artifact_sha256=hash_file(p))
            save_json(audit/f'{stem}.json',dict(r,contexts=cm)); results.append(r)
    screen={}
    for method in MS[2:]:
        pairs=[(next(r for r in results if r['method']==method and r['seed']==s),
                next(r for r in results if r['method']=='heuristic' and r['seed']==s)) for s in AS]
        records = [dict(seed=r['seed'], median_error_reduction=1-r['weighted_median_error']/h['weighted_median_error'],
                        screen_pass=bool(1-r['weighted_median_error']/h['weighted_median_error'] >= .30 and
                                         r['all_zero_distance_fraction']-h['all_zero_distance_fraction'] <= .001))
                   for r,h in pairs]
        screen[method]=dict(seeds=records, all_seed_screen_pass=all(r['screen_pass'] for r in records))
    save_json(audit/'SUMMARY.json',dict(results=results,mechanism_screen=screen,selection_sha256=selection_hash,selection_unchanged=True))
    save_json(audit/'COMPLETE.json',dict(summary_sha256=hash_file(audit/'SUMMARY.json')))
    return inp,cal,audit


def verify(bundle):
    return v.verify(*bundle,fixture=True)


def mutate_npz(path, change):
    with np.load(path,allow_pickle=False) as z:
        values={k:z[k] for k in z.files}
    change(values)
    np.savez(path,**values)


def rebind_cal(bundle,g):
    _,cal,audit=bundle
    s=json.loads((cal/'SELECTION.json').read_text())
    s['records'][g]['sha256']=hash_file(cal/f'context_{g:03d}.npz')
    save_json(cal/'SELECTION.json',s)
    h=hash_file(cal/'SELECTION.json')
    save_json(cal/'COMPLETE.json',dict(selection_sha256=h))
    summary=json.loads((audit/'SUMMARY.json').read_text()); summary['selection_sha256']=h
    save_json(audit/'SUMMARY.json',summary)
    save_json(audit/'COMPLETE.json',dict(summary_sha256=hash_file(audit/'SUMMARY.json')))


def test_correct_complete_fixture_and_cli(bundle,tmp_path):
    r=verify(bundle)
    assert r['counts']['loo_rows']==110
    assert r['counts']['audit_rows']==1320
    assert r['counts']['audit_files']==12
    assert r['counts']['audit_contexts']==48
    assert r['counts']['inverse_rows']==r['counts']['calibration_rows']
    out=tmp_path/'REPORT.json'
    command=[sys.executable,str(Path(v.__file__)),'--inputs',str(bundle[0]),'--calibration',str(bundle[1]),
             '--audit',str(bundle[2]),'--output',str(out),'--fixture']
    result=subprocess.run(command,text=True,capture_output=True)
    (tmp_path/'cli.stdout.log').write_text(result.stdout)
    (tmp_path/'cli.stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    assert json.loads(out.read_text())['status']=='verified'
    original=out.read_bytes()
    second=subprocess.run(command,text=True,capture_output=True)
    (tmp_path/'overwrite.stderr.log').write_text(second.stderr)
    assert second.returncode!=0 and out.read_bytes()==original


def test_default_refuses_small_population(bundle):
    with pytest.raises(v.ValidationError,match='default population'):
        v.verify(*bundle)


def test_nan_refused_before_zero_fallback(bundle):
    p=bundle[1]/'context_001.npz'
    mutate_npz(p,lambda z:z['quantiles'].__setitem__((0,0,0),np.nan))
    rebind_cal(bundle,1)
    with pytest.raises(v.ValidationError,match='nonfinite'):
        verify(bundle)


@pytest.mark.parametrize('where',['calibration','audit','inverse'])
def test_altered_distance_refused(bundle,where):
    if where=='audit':
        p=bundle[2]/'heuristic_43017.npz'
        mutate_npz(p,lambda z:z['distances'].__setitem__(0,z['distances'][0]+.1))
    else:
        p=bundle[1]/'context_000.npz'
        name='inverse_distances' if where=='inverse' else 'distances'
        mutate_npz(p,lambda z:z[name].__setitem__((0,0,0),z[name][0,0,0]+.1))
        rebind_cal(bundle,0)
    with pytest.raises(v.ValidationError,match='distances'):
        verify(bundle)


def test_altered_marginal_refused(bundle):
    p=bundle[1]/'context_000.npz'
    mutate_npz(p,lambda z:z['tables'].__setitem__((0,0,0,0),9999))
    rebind_cal(bundle,0)
    with pytest.raises(v.ValidationError,match='marginal'):
        verify(bundle)


def test_altered_selection_refused(bundle):
    p=bundle[1]/'SELECTION.json'; s=json.loads(p.read_text())
    s['records'][0]['empirical_median']=.5
    save_json(p,s); rebind_cal(bundle,0)
    with pytest.raises(v.ValidationError,match='selected E'):
        verify(bundle)


def test_missing_audit_seed_refused(bundle):
    (bundle[2]/'unit_beta_43041.npz').unlink()
    with pytest.raises(FileNotFoundError):
        verify(bundle)


@pytest.mark.parametrize('mode',['duplicate','misaligned'])
def test_bad_audit_contexts_refused(bundle,mode):
    p=bundle[2]/'unit_beta_43017.npz'
    def damage(z):
        z['contexts'][1]=z['contexts'][0] if mode=='duplicate' else z['contexts'][2]
    mutate_npz(p,damage)
    with pytest.raises(v.ValidationError,match='contexts'):
        verify(bundle)


def test_input_label_context_mismatch(bundle):
    p=bundle[0]/'h001-seed17/run/data/train.npy'
    t=np.load(p,allow_pickle=False); t[0,12]=99; np.save(p,t)
    with pytest.raises(v.ValidationError,match='context identity'):
        verify(bundle)


def test_selection_rules_and_ties():
    w=np.array([1.,2.,4.]); target=np.array([1.,2.,3.,4.,5.])
    observed=np.array([[target+2,target,target+1],[target+2,target,target+1]])
    assert v.choose(w,observed,target,'median')==2
    assert v.choose(w,observed,target,'quantile')==2
    observed[:]=target
    assert v.choose(w,observed,target,'median')==1
    observed[0,2,0]=np.nan
    with pytest.raises(v.ValidationError,match='nonfinite'):
        v.choose(w,observed,np.zeros(5),'median')


def test_write_serialization_and_exclusive(tmp_path):
    p=tmp_path/'new.json'
    with pytest.raises(ValueError):
        v.write_new(p,{'bad':float('nan')})
    assert not p.exists() and not list(tmp_path.glob('*.tmp'))
    v.write_new(p,{'ok':True})
    with pytest.raises(FileExistsError):
        v.write_new(p,{'ok':False})
    assert json.loads(p.read_text())=={'ok':True}


def test_malformed_shape_refused(bundle):
    p=bundle[1]/'context_000.npz'
    mutate_npz(p,lambda z:z.__setitem__('distances',z['distances'][:,:,:1]))
    rebind_cal(bundle,0)
    with pytest.raises(v.ValidationError,match='shape'):
        verify(bundle)


def test_failed_cli_retains_partial_log_and_no_report(bundle,tmp_path):
    p=bundle[2]/'empirical_quantile_43041.npz'
    mutate_npz(p,lambda z:z['distances'].__setitem__(-1, .125))
    output=tmp_path/'FAILED_REPORT.json'
    command=[sys.executable,str(Path(v.__file__)),'--inputs',str(bundle[0]),'--calibration',str(bundle[1]),
             '--audit',str(bundle[2]),'--output',str(output),'--fixture']
    result=subprocess.run(command,text=True,capture_output=True)
    (tmp_path/'failure.stdout.log').write_text(result.stdout)
    (tmp_path/'failure.stderr.log').write_text(result.stderr)
    save_json(tmp_path/'failure.exit.json',dict(command=command,exit_code=result.returncode))
    assert result.returncode!=0 and not output.exists()
    assert 'audit heuristic_43041 all 110 rows verified' in result.stderr
    assert 'ValidationError: audit distances: mismatch' in result.stderr


def test_summary_screen_tamper(bundle):
    p=bundle[2]/'SUMMARY.json'
    summary=json.loads(p.read_text())
    summary['mechanism_screen']['empirical_median']['all_seed_screen_pass']=not summary['mechanism_screen']['empirical_median']['all_seed_screen_pass']
    save_json(p,summary)
    save_json(bundle[2]/'COMPLETE.json',dict(summary_sha256=hash_file(p)))
    with pytest.raises(v.ValidationError,match='mechanism screen'):
        verify(bundle)


def test_missing_context_list_refused_explicitly(bundle):
    p=bundle[2]/'unit_beta_43017.json'
    saved=json.loads(p.read_text()); del saved['contexts']; save_json(p,saved)
    with pytest.raises(v.ValidationError,match='missing fields'):
        verify(bundle)


def rewrite_summary(bundle, change):
    p = bundle[2]/'SUMMARY.json'
    summary = json.loads(p.read_text())
    change(summary)
    save_json(p, summary)
    save_json(bundle[2]/'COMPLETE.json', dict(summary_sha256=hash_file(p)))


@pytest.mark.parametrize('order', ['seed_method', 'reverse'])
def test_summary_permutation_accepted(bundle, order):
    def change(s):
        s['results'] = sorted(s['results'], key=lambda r: (r['seed'], MS.index(r['method']))) if order == 'seed_method' else s['results'][::-1]
        for screen in s['mechanism_screen'].values():
            screen['seeds'].reverse()
    rewrite_summary(bundle, change)
    assert verify(bundle)['status'] == 'verified'


@pytest.mark.parametrize('mode', ['duplicate_replace', 'duplicate_append', 'missing', 'extra'])
def test_summary_identity_rejected(bundle, mode):
    def change(s):
        rows = s['results']
        if mode == 'duplicate_replace': rows[-1] = dict(rows[0])
        elif mode == 'duplicate_append': rows.append(dict(rows[0]))
        elif mode == 'missing': rows.pop()
        else: rows.append(dict(rows[0], seed=99999))
    rewrite_summary(bundle, change)
    with pytest.raises(v.ValidationError, match='summary results:.*(duplicate|missing|unexpected)'):
        verify(bundle)


@pytest.mark.parametrize('method', MS[2:])
@pytest.mark.parametrize('mode', ['reduction', 'flag', 'all_flag', 'missing', 'extra', 'duplicate_replace', 'duplicate_append'])
def test_nested_screen_corruption_rejected(bundle, method, mode):
    def change(s):
        screen = s['mechanism_screen'][method]
        records = screen['seeds']
        if mode == 'reduction': records[1]['median_error_reduction'] += .01
        elif mode == 'flag': records[1]['screen_pass'] = not records[1]['screen_pass']
        elif mode == 'all_flag': screen['all_seed_screen_pass'] = not screen['all_seed_screen_pass']
        elif mode == 'missing': records.pop()
        elif mode == 'extra': records.append(dict(records[0], seed=99999))
        elif mode == 'duplicate_replace': records[-1] = dict(records[0])
        else: records.append(dict(records[0]))
    rewrite_summary(bundle, change)
    with pytest.raises(v.ValidationError, match='mechanism screen'):
        verify(bundle)


def test_screen_fixed_thresholds():
    rows = [dict(method=m, seed=s, weighted_median_error=1., all_zero_distance_fraction=0.) for m in MS for s in AS]
    for row in rows:
        if row['method'] in MS[2:]:
            row.update(weighted_median_error=.7, all_zero_distance_fraction=.001)
    assert all(x['all_seed_screen_pass'] for x in v.screen(rows).values())
    rows[-1]['weighted_median_error'] = .70000001
    assert not v.screen(rows)['empirical_quantile']['all_seed_screen_pass']
    rows[-1]['weighted_median_error'] = .7
    rows[-1]['all_zero_distance_fraction'] = .00100001
    assert not v.screen(rows)['empirical_quantile']['all_seed_screen_pass']
