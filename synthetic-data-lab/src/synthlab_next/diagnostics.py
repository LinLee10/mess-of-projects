"""Intrinsic diagnostics stay separate from downstream utility and privacy claims."""
from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from .common import validate_xy


def table_diagnostics(real_x,real_y,synthetic_x,synthetic_y):
    real_x,real_y=validate_xy(real_x,real_y);synthetic_x,synthetic_y=validate_xy(synthetic_x,synthetic_y)
    if real_x.shape[1]!=synthetic_x.shape[1]:raise ValueError('Schema mismatch')
    scale=np.maximum(real_x.std(0),1e-8)
    marginal=float(np.mean([wasserstein_distance(real_x[:,j]/scale[j],synthetic_x[:,j]/scale[j]) for j in range(real_x.shape[1])]))
    valid=(real_x.std(0)>1e-8)&(synthetic_x.std(0)>1e-8)
    correlation=None
    if valid.sum()>1:
        a=np.corrcoef(real_x[:,valid],rowvar=False);b=np.corrcoef(synthetic_x[:,valid],rowvar=False)
        correlation=float(np.mean(np.abs(a[np.triu_indices_from(a,1)]-b[np.triu_indices_from(b,1)])))
    real_keys={(int(y),row.tobytes()) for row,y in zip(real_x,real_y)}
    generated_keys=[(int(y),row.tobytes()) for row,y in zip(synthetic_x,synthetic_y)]
    d=NearestNeighbors(n_neighbors=1).fit(real_x/scale).kneighbors(synthetic_x/scale,return_distance=True)[0][:,0]
    return {'n':len(synthetic_y),'marginal_wasserstein_train_standardized':marginal,
        'correlation_mean_absolute_error_train':correlation,'correlation_columns':int(valid.sum()),
        'exact_training_copy_rate':float(np.mean([k in real_keys for k in generated_keys])),
        'exact_internal_duplicate_rate':1-len(set(generated_keys))/len(generated_keys),
        'nearest_training_distance_median':float(np.median(d)),
        'negative_feature_row_fraction':float(np.mean(np.any(synthetic_x<0,axis=1))),
        'class_counts':np.bincount(synthetic_y).tolist(),
        'limitations':'Training reference diagnostics favor memorization; distances do not establish privacy or semantic validity'}
