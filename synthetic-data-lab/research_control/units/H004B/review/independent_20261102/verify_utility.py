"""Independent H004B artifact acceptance arithmetic and inference checker.
No production imports. Full execution requires numpy and LightGBM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = (43017, 43029, 43041)
BASE = ('unit_beta', 'heuristic', 'empirical_median', 'empirical_quantile')
METHODS = BASE + ('iid_beta_inverse', 'iid_beta_rank')
CONDITIONS = {('real', 0)} | {(m, s) for m in METHODS for s in SEEDS}
UNITS = Path(__file__).resolve().parents[1] / 'synthetic-data-lab/research_control/units'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key')
            d[k] = v
        return d
    def invalid(x):
        raise ValueError('nonfinite JSON constant')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def finite(x, shape):
    a = np.asarray(x)
    require(a.shape == shape and a.dtype.kind in 'iuf' and np.isfinite(a).all(), 'shape/nonfinite numeric array')
    return a


def equal(actual, expected):
    """Compare complete JSON structures, with strict shapes and arithmetic tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'object fields')
        for k in expected:
            equal(actual[k], expected[k])
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected, 'boolean mismatch')
    elif isinstance(expected, (int, np.integer)):
        require(type(actual) is int and actual == expected, 'integer mismatch')
    else:
        a = finite(actual, np.asarray(expected).shape)
        if np.asarray(expected).dtype.kind in 'iu':
            require(a.dtype.kind in 'iu' and np.array_equal(a, expected), 'integer array mismatch')
        require(np.allclose(a, expected, rtol=0, atol=1e-14), 'arithmetic mismatch')


def table(a, n):
    require(a.shape == (n, 13) and a.dtype.kind in 'iu', 'compact table schema')
    for col, size in ((10, 4), (11, 40), (12, 7)):
        require(((a[:, col] >= 0) & (a[:, col] < size)).all(), 'category range')
    return a


def probabilities(p, n):
    p = finite(p, (n, 7))
    require((p >= 0).all() and (p <= 1).all(), 'probability range')
    require(np.allclose(p.sum(1), 1, rtol=0, atol=1e-10), 'probability normalization')
    return p


def score(y, p):
    require(y.ndim == 1 and y.dtype.kind in 'iu' and len(y) > 0, 'label shape/type')
    require(((y >= 0) & (y < 7)).all(), 'label range')
    pred = probabilities(p, len(y)).argmax(1)
    cm = np.zeros((7, 7), dtype=np.int64)
    for c in range(7):
        cm[c] = np.bincount(pred[y == c], minlength=7)
    require((cm.sum(1) > 0).all(), 'missing development class')
    recall = cm.diagonal() / cm.sum(1)
    return dict(n=len(y), correct=int(cm.trace()), accuracy=float(cm.trace()/len(y)),
                balanced_accuracy=float(recall.mean()), class_recall=recall.tolist(), confusion_matrix=cm.tolist())


def compare_inference(saved, loaded):
    probabilities(saved, len(saved)); probabilities(loaded, len(saved))
    require(np.allclose(saved, loaded, rtol=1e-9, atol=1e-10), 'model probability mismatch')
    require(np.array_equal(saved.argmax(1), loaded.argmax(1)), 'model classes changed')


def linear_inference(model, dev, training):
    mean = finite(model['mean'], (10,)); scale = finite(model['scale'], (10,))
    require((scale > 0).all(), 'nonpositive scale')
    require(np.array_equal(mean, training[:, :10].mean(0)), 'scaling mean binding')
    require(np.array_equal(scale, np.maximum(training[:, :10].std(0), 1e-8)), 'scaling std binding')
    require(model['classes'] == list(range(7)), 'linear classes')
    require(model['converged'] is True and model['warnings'] == [], 'linear convergence/warnings')
    it = finite(model['iterations'], (1,))
    require(it.dtype.kind in 'iu' and 0 < it[0] < 1000, 'iteration ceiling')
    require(model['configuration'] == {'C':1., 'solver':'lbfgs', 'max_iter':1000, 'tol':1e-5}, 'linear configuration')
    coef = finite(model['coef'], (7, 54)); intercept = finite(model['intercept'], (7,))
    x = np.concatenate(((dev[:, :10]-mean)/scale, np.eye(4)[dev[:, 10]], np.eye(40)[dev[:, 11]]), axis=1)
    logits = x @ coef.T + intercept
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def manifest(root, name, exclusions=(), complete=False):
    root = Path(root).resolve()
    mapping = read(root/name)
    require(isinstance(mapping, dict) and bool(mapping), 'empty manifest')
    for rel, digest in mapping.items():
        p = root/rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'unsafe manifest path')
        require(p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink(), 'unsafe artifact')
        require(all(not q.is_symlink() for q in p.parents if q != root.parent), 'symlink ancestor')
        require(sha(p) == digest, 'artifact hash mismatch: '+rel)
    if complete:
        found = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        require(set(mapping) == found-set(exclusions), 'manifest coverage')
    return mapping


def matrix(results):
    require(isinstance(results, list) and len(results) == 19, '19 conditions required')
    keys = [(r['method'], r['seed']) for r in results]
    require(len(set(keys)) == 19 and set(keys) == CONDITIONS, 'condition matrix mismatch')
    return dict(zip(keys, results))


def engineering(results, geometry):
    rows = matrix(results); output = {}
    for method in ('empirical_median', 'empirical_quantile'):
        diffs = {}; recalls = {}
        for learner in ('tree', 'linear'):
            diffs[learner] = float(np.mean([rows[method,s][learner]['balanced_accuracy']-rows['heuristic',s][learner]['balanced_accuracy'] for s in SEEDS]))
            recalls[learner] = np.mean([np.asarray(rows[method,s][learner]['class_recall'])-rows['heuristic',s][learner]['class_recall'] for s in SEEDS], axis=0).tolist()
        g = geometry[method]
        if isinstance(g, dict):
            g = g['all_seed_screen_pass']
        require(type(g) is bool, 'geometry screen boolean')
        output[method] = dict(mean_balanced_accuracy_change=diffs, mean_class_recall_change=recalls,
            geometry_screen=g, exploratory_engineering_screen_pass=bool(g and diffs['tree'] >= -.01 and diffs['linear'] >= -.005 and min(recalls['tree']) >= -.03 and min(recalls['linear']) >= -.03))
    return output


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def tie_keys(ids, j):
    # Frozen identity hash, independently expressed using Python integers.
    mask = (1 << 64)-1
    out = []
    for identity in ids:
        z = ((int(identity) ^ ((j+1)*7919) ^ 217) + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30))*0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27))*0x94D049BB133111EB) & mask
        out.append(z ^ (z >> 31))
    return np.array(out, dtype=np.uint64)


def references(train, groups, seed, run):
    n = len(train); lineage = load_npz(run/f'paper_lineage_{seed}.npz')
    u = finite(lineage['latent'], (n,10))
    require(((u >= 0) & (u <= 1)).all(), 'latent bounds')
    require(np.array_equal(lineage['contexts'], groups['contexts']) and np.array_equal(lineage['counts'], groups['counts']), 'lineage contexts')
    require(np.array_equal(lineage['original_ids'], groups['group_indices']), 'lineage original ids')
    require(lineage['anchor_ids'].shape == (n,) and lineage['anchor_ids'].dtype.kind in 'iu', 'anchor shape')
    inverse = np.empty_like(train); rank = np.empty_like(train)
    for g, count in enumerate(groups['counts']):
        a,b = groups['offsets'][g:g+2]; ids = groups['group_indices'][a:b]; x = train[ids,:10]
        r = np.empty(x.shape, dtype=np.int64)
        for j in range(10):
            order = np.lexsort((tie_keys(ids,j), x[:,j])); r[order,j] = np.arange(1,count+1)
        rng = np.random.default_rng(np.random.SeedSequence([seed,g,3202]))
        anchors = rng.integers(0,count,size=count)
        latent = rng.beta(r[anchors],count+1-r[anchors])
        require(np.array_equal(lineage['anchor_ids'][a:b],ids[anchors]) and np.array_equal(u[a:b],latent), 'anchor/beta reconstruction')
        sorted_x = np.sort(x,axis=0)
        inverse[a:b,:10] = sorted_x[np.minimum((count*latent).astype(np.int64),count-1),np.arange(10)]
        for j in range(10):
            order = np.lexsort((tie_keys(np.arange(count),j),latent[:,j]))
            rank[a+order,j] = sorted_x[:,j]
        inverse[a:b,10:] = rank[a:b,10:] = groups['contexts'][g]
    for method, expected in (('iid_beta_inverse',inverse),('iid_beta_rank',rank)):
        z = load_npz(run/f'{method}_{seed}.npz'); table(z['table'],n)
        require(np.array_equal(z['table'],expected), 'paired reference reconstruction')
        require(np.array_equal(z['counts'],groups['counts']) and np.array_equal(z['contexts'],groups['contexts']), 'reference metadata')
    return {'iid_beta_inverse':inverse,'iid_beta_rank':rank}


def verify(inputs, audit, verification, run):
    # LightGBM is deliberately optional for helper controls.
    import lightgbm as lgb
    inputs,audit,verification,run = map(lambda p:Path(p).resolve(strict=True),(inputs,audit,verification,run))
    before = {str(p):sha(p) for root in (inputs,audit,run) for p in root.rglob('*') if p.is_file()}
    before[str(verification)] = sha(verification)
    for folder in ('h001-seed17','b002-seed17','h003-seed17'):
        manifest(inputs/folder,'ARTIFACT_SHA256.json')
    manifest(run,'MANIFEST.json',('MANIFEST.json','COMPLETE.json'),True)
    done = read(run/'COMPLETE.json')
    require(done == {'status':'completed','manifest_sha256':sha(run/'MANIFEST.json'),'fresh_student_fits':38}, 'COMPLETE seal')
    report = read(verification)
    require(report['status']=='verified' and report['fixture'] is False, 'H004A prerequisite')
    for k,v in dict(training_rows=348563,contexts=232,audit_files=12,audit_rows=4182756).items():
        require(type(report['counts'][k]) is int and report['counts'][k]==v,'H004A population')
    bindings = report['artifact_sha256']
    for p,h in bindings.items():
        require(Path(p).is_absolute() and sha(p)==h,'H004A evidence identity')
    trainpath = inputs/'h001-seed17/run/data/train.npy'; devpath = inputs/'h001-seed17/run/data/dev.npy'
    require(sha(trainpath)=='c39bf8a33ec369f8472622623251d1c4aa9082ce4a582397c311599f8f653418','frozen training identity')
    train = table(np.load(trainpath,allow_pickle=False),348563); dev = table(np.load(devpath,allow_pickle=False),116314)
    require(train.dtype==np.dtype('int64') and dev.dtype==np.dtype('int64'),'full table dtypes')
    groups = load_npz(inputs/'b002-seed17/run/generator.npz')
    for key in ('contexts','counts','offsets','group_indices','sorted'):
        require(groups[key].dtype.kind in 'iu', 'integer context metadata')
    require(groups['contexts'].shape==(232,3) and groups['counts'].shape==(232,), 'context dimensions')
    require(np.array_equal(groups['offsets'],np.r_[0,np.cumsum(groups['counts'])]) and (groups['counts']>0).all(),'context offsets')
    require(np.array_equal(np.sort(groups['group_indices']),np.arange(len(train))),'context membership')
    require(str(groups['input_sha256'].item())==hashlib.sha256(train.tobytes()).hexdigest(),'source-table binding')
    for g in range(232):
        a,b = groups['offsets'][g:g+2]; ids=groups['group_indices'][a:b]
        require(np.all(train[ids,10:]==groups['contexts'][g]),'source context binding')
        require(np.array_equal(np.sort(train[ids,:10],axis=0),groups['sorted'][a:b]),'source marginals')
    start = read(run/'START.json')
    require(start['generation_seeds']==list(SEEDS) and start['fit_count_planned']==38 and start['reserved_data_access'] is False,'START plan')
    require(start['h004a_verification_sha256']==sha(verification) and start['development_sha256']==sha(devpath),'START identities')
    for key,rel in dict(training='h001-seed17/run/data/train.npy',contexts='b002-seed17/run/generator.npz',geometry='h003-seed17/run/geometry.npz').items():
        require(start['inputs'][key]=={'path':str(inputs/rel),'sha256':sha(inputs/rel)},'protected input identity')
    expected_sources = {p.name:sha(p) for p in (UNITS/'H004B/src').glob('*.py')}
    require(start['source_sha256']==expected_sources,'runner source binding')
    require(set(start['dependency_sha256'])=={'H003R/review/audit_recovery.py','H001/src/run_h001.py','H001C/src/second_learner.py','B002/src/run.py'},'dependency coverage')
    for rel,h in start['dependency_sha256'].items():
        require(sha(UNITS/rel)==h,'dependency source binding')
    order = np.load(run/'student_row_order.npy',allow_pickle=False)
    require(np.array_equal(order,np.random.default_rng(88017).permutation(len(train))) and order.dtype.kind in 'iu','permutation88017')
    summary = read(run/'SUMMARY.json'); rows = matrix(summary['results'])
    require(summary['fresh_student_fits']==38 and summary['reserved_scored'] is False and summary['scientific_novelty_established'] is False,'SUMMARY declarations')
    model_files = {str(p.relative_to(run)) for p in run.rglob('*') if p.name in ('model.txt','model.json')}
    expected_models = {f'{m}_{s}/{l}/{f}' for m,s in CONDITIONS for l,f in (('tree','model.txt'),('linear','model.json'))}
    require(model_files==expected_models and len(model_files)==38,'38 model files')
    for seed in (0,)+SEEDS:
        refs = {} if seed==0 else references(train,groups,seed,run)
        for method in (('real',) if seed==0 else METHODS):
            folder=run/f'{method}_{seed}'; result=read(folder/'RESULT.json')
            require(result==rows[method,seed],'RESULT/SUMMARY disagreement')
            require(result['training_rows']==len(train) and result['development_rows']==len(dev),'result populations')
            if method=='real': raw=train
            elif method in refs: raw=refs[method]
            else:
                p=audit/f'{method}_{seed}.npz'
                require(str(p) in bindings and sha(p)==bindings[str(p)],'consumed audit binding')
                raw=load_npz(p)['table']; table(raw,len(train))
            training=raw[order].astype(np.int64); h=hashlib.sha256(training.tobytes()).hexdigest()
            require(result['student_table_sha256']==h,'permuted student table hash')
            for learner in ('tree','linear'):
                dest=folder/learner; z=load_npz(dest/'predictions.npz')
                require(np.array_equal(z['labels'],dev[:,12]) and z['labels'].shape==(len(dev),) and z['labels'].dtype.kind in 'iu','saved development labels')
                arithmetic=score(z['labels'],z['probabilities']); metrics=read(dest/'METRICS.json')
                equal(metrics['utility'],arithmetic); equal(result[learner],arithmetic)
                require(metrics['training_rows']==len(train),'learner training rows')
                require(np.isfinite(metrics['fit_and_predict_seconds']) and metrics['fit_and_predict_seconds']>=0,'fit time')
                if learner=='linear':
                    require(metrics['input_sha256']==h,'linear training hash')
                    loaded=linear_inference(read(dest/'model.json'),dev,training)
                else:
                    seal=read(dest/'DONE.json'); require(seal['status']=='completed','tree seal')
                    for name,digest in seal['files'].items(): require(sha(dest/name)==digest,'tree DONE hash')
                    require(set(seal['files'])=={'predictions.npz','model.txt','METRICS.json'},'tree DONE coverage')
                    require(metrics['config']=={'n_estimators':150,'num_leaves':63,'learning_rate':.08,'max_depth':-1,'min_child_samples':30,'subsample':1.,'colsample_bytree':1.,'n_jobs':2,'random_state':991,'verbosity':-1,'deterministic':True,'force_col_wise':True},'tree recipe metadata')
                    require(metrics['generation_data_hash']==h and metrics['generation_seed']==seed,'tree training binding')
                    model=lgb.Booster(model_file=str(dest/'model.txt'))
                    require(model.num_feature()==12 and model.num_model_per_iteration()==7 and model.current_iteration()==150,'tree model dimensions')
                    loaded=model.predict(dev[:,:12],num_threads=2)
                compare_inference(z['probabilities'],loaded)
    reconstructed=engineering(summary['results'],report['mechanism_screen'])
    equal(summary['engineering_screen'],reconstructed)
    require(all(sha(p)==h for p,h in before.items()),'protected artifacts changed during verification')
    return dict(status='verified_supported_checks',conditions=19,model_files=38,engineering_screen=reconstructed,
                protected_sha256=before,unsupported_checks=['independent iid nearest-distance/context-geometry/dependence diagnostic arithmetic','H004A geometry screen numerical rederivation (relies on bound prerequisite report)','fit history and live runtime/resource observations','tree training recipe cannot be proven solely by model reload','historical checker checked subrecords are retained but not interpreted','protocol and runtime package identity are recorded locally but not independently authenticated from runner START'],
                scientific_endorsement=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','audit','verification','run','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(),'output must be new')
    for source in (args.inputs,args.audit,args.run):
        require(not args.output.resolve().is_relative_to(source.resolve()),'output beneath protected source')
    require(args.output.resolve()!=args.verification.resolve(),'output overwrites prerequisite')
    try:
        result=verify(args.inputs,args.audit,args.verification,args.run)
    except Exception as exc:
        with args.output.open('x') as f: json.dump(dict(status='failed',error_type=type(exc).__name__,error=str(exc)),f,indent=2,allow_nan=False)
        raise
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__': main()
