"""Contextual empirical Gaussian copula reference. Not a novelty claim."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf


def validate(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 13 or a.dtype.kind not in 'iu' or len(a) < 1:
        raise ValueError('Expected nonempty integer Covertype table')
    for j, k in ((10, 4), (11, 40), (12, 7)):
        if np.any((a[:, j] < 0) | (a[:, j] >= k)):
            raise ValueError('Illegal context')
    return a


def data_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ContextCopula:
    def fit(self, table):
        table = validate(table)
        self.contexts, inv, self.counts = np.unique(table[:, 10:], axis=0, return_inverse=True, return_counts=True)
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.sorted = np.empty((len(table), 10), dtype=np.int64)
        self.correlation = np.empty((len(self.counts), 10, 10))
        self.shrinkage = np.zeros(len(self.counts))
        self.input_sha256 = data_digest(table)
        self.group_indices = np.argsort(inv, kind='stable')
        self.latent = np.empty((len(table), 10), dtype=float)
        for g, n in enumerate(self.counts):
            sl = slice(self.offsets[g], self.offsets[g+1])
            rows = table[self.group_indices[sl], :10]
            self.sorted[sl] = np.sort(rows, axis=0)
            z = np.column_stack([ndtri((rankdata(rows[:, j], method='average') - .5) / n) for j in range(10)])
            self.latent[sl] = z
            varying = rows.std(0) > 0
            corr = np.eye(10)
            if n > 1 and varying.any():
                a = z[:, varying]
                a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)
                estimate = LedoitWolf(assume_centered=True).fit(a)
                cov = estimate.covariance_
                sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
                c = cov / sd[:, None] / sd[None, :]
                c = (c + c.T) / 2
                c = .999999 * c + .000001 * np.eye(len(c))
                corr[np.ix_(varying, varying)] = c
                self.shrinkage[g] = estimate.shrinkage_
            self.correlation[g] = corr
        return self

    def draw(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Nonnegative integer seed required')
        rng = np.random.default_rng(seed)
        result = np.empty((self.offsets[-1], 13), dtype=np.int64)
        for g, n in enumerate(self.counts):
            sl = slice(self.offsets[g], self.offsets[g+1])
            noise = rng.standard_normal((n, 10))
            z = noise @ np.linalg.cholesky(self.correlation[g]).T
            u = ndtr(z)
            reference = self.sorted[sl]
            for j in range(10):
                result[sl, j] = np.rint(np.interp(u[:, j] * (n-1), np.arange(n), reference[:, j]))
            result[sl, 10:] = self.contexts[g]
        order = rng.permutation(len(result))
        return result[order], order

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise ValueError('Refuse model overwrite')
        np.savez_compressed(path, contexts=self.contexts, counts=self.counts,
                            offsets=self.offsets, sorted=self.sorted,
                            correlation=self.correlation, shrinkage=self.shrinkage,
                            group_indices=self.group_indices, latent=self.latent,
                            input_sha256=np.array(self.input_sha256))

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as f:
            for key in ('contexts','counts','offsets','sorted','correlation','shrinkage','group_indices','latent'):
                setattr(model, key, f[key])
            model.input_sha256 = str(f['input_sha256'].item())
        if not np.isfinite(model.correlation).all() or not np.isfinite(model.latent).all():
            raise ValueError('Nonfinite model')
        if model.offsets[-1] != len(model.sorted) or model.correlation.shape != (len(model.counts),10,10):
            raise ValueError('Corrupt model shapes')
        return model
