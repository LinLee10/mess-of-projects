"""Exploratory real versus real distance control; no fitting or reserved data."""
import json
from pathlib import Path
import numpy as np
from context_proximity import distances_summary

def run(reference: Path, generated: Path, output: Path) -> None:
    dev = np.load(reference/'run/data/dev.npy', allow_pickle=False)
    train = np.load(reference/'run/data/train.npy', allow_pickle=False)
    with np.load(generated/'run/real_development_distances.npz', allow_pickle=False) as z:
        distances = z['distance']
    assert len(dev) == len(distances) == 116314
    contexts, counts = np.unique(train[:, 10:], axis=0, return_counts=True)
    rows = []
    for i, (context, count) in enumerate(zip(contexts, counts)):
        idx = np.flatnonzero(np.all(dev[:, 10:] == context, axis=1))
        if len(idx) < 100:
            continue
        shuffled = np.random.default_rng(60617+i).permutation(idx)
        a, b = np.array_split(shuffled, 2)
        stats = distances_summary(distances[a], distances[b])
        rows.append({'context': context.tolist(), 'training_rows': int(count), 'development_rows': len(idx), 'stats': stats})
    total = sum(row['training_rows'] for row in rows)
    result = {'status': 'completed_exploratory', 'final_test_scored': False, 'permutation_seed_base': 60617,
              'minimum_total_development_rows': 100, 'contexts': len(rows), 'covered_training_rows': total,
              'weighted_ks': sum(row['training_rows']*row['stats']['ks_distance'] for row in rows)/total,
              'weighted_fraction_below_reference_median': sum(row['training_rows']*row['stats']['synthetic_below_reference_quantiles'][3] for row in rows)/total,
              'weighted_fraction_below_reference_q05': sum(row['training_rows']*row['stats']['synthetic_below_reference_quantiles'][1] for row in rows)/total,
              'rows': rows,
              'limitations': 'One fixed random split of existing development distances. Descriptive finite sample control, not independent external validation.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
