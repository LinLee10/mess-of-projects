"""Measured beta bandwidth calibration. Established components, no novelty claim.

This module is deliberately independent of development labels and data loaders.
The full experiment wrapper pins the archived training inputs and dependencies.
"""
from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

_H002 = Path(__file__).resolve().parents[2] / 'H002' / 'src'
if str(_H002) not in sys.path:
    sys.path.insert(0, str(_H002))
from rank_synthesis import assign_columns, ordinal

QUANTILES = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integral scalar, not bool')
    if value < minimum:
        raise ValueError(f'{name} is below {minimum}')
    return int(value)


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _finite_array(value, name: str, ndim: int) -> np.ndarray:
    try:
        a = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if a.ndim != ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite real array of dimension {ndim}')
    return a


def candidate_widths(n: int, heuristic: float) -> np.ndarray:
    n = _integer(n, 'n', 1)
    h = _real(heuristic, 'heuristic')
    if not 1 <= h <= n:
        raise ValueError('Heuristic must be within [1,n]')
    # Multiplication cannot overflow because n bounds h and n is a row count.
    return np.unique(np.clip([1.0, h / 2, h, 2 * h, 4 * h, 8 * h], 1.0, float(n)))


def choose_width(widths, observed, target, objective: str = 'median') -> float:
    w = _finite_array(widths, 'widths', 1)
    q = _finite_array(observed, 'observed', 3)
    t = _finite_array(target, 'target', 1)
    if not len(w) or np.any(w < 1) or np.any(np.diff(w) <= 0):
        raise ValueError('Widths must be nonempty, increasing and at least one')
    if t.shape != (5,) or q.shape[0] < 1 or q.shape[1:] != (len(w), 5):
        raise ValueError('Exact quantile and width shapes are required')
    if np.any(t < 0) or np.any(q < 0) or np.any(np.diff(t) < 0) or np.any(np.diff(q, axis=2) < 0):
        raise ValueError('Distance quantiles must be nonnegative and nondecreasing')
    if objective not in ('median', 'quantile'):
        raise ValueError('Unknown calibration objective')
    if t[2] == 0:
        return float(w[0])
    if objective == 'median':
        loss = np.mean(np.abs(q[:, :, 2] - t[2]) / t[2], axis=0)
    else:
        loss = np.mean(np.abs(q - t[None, None, :]) / t[2], axis=(0, 2))
    if not np.isfinite(loss).all():
        raise ValueError('Nonfinite objective')
    return float(w[int(np.argmin(loss))])


def sample_context(reference, ids, width, seed, context_index=0) -> dict[str, np.ndarray]:
    x = _finite_array(reference, 'reference', 2)
    i = _finite_array(ids, 'ids', 1)
    if x.dtype.kind not in 'iu' or min(x.shape) < 1 or i.dtype.kind not in 'iu':
        raise ValueError('Nonempty integer reference and integer ids required')
    if i.shape != (len(x),) or np.any(i < 0) or len(np.unique(i)) != len(i):
        raise ValueError('Nonnegative unique aligned ids required')
    if np.any(x > np.iinfo(np.int64).max) or np.any(x < np.iinfo(np.int64).min):
        raise ValueError('Reference outside int64 representation')
    f = _real(width, 'width')
    if not 1 <= f <= len(x):
        raise ValueError('Bandwidth must be within [1,n]')
    s = _integer(seed, 'seed')
    c = _integer(context_index, 'context_index')
    r = np.column_stack([ordinal(x[:, j], i, j) for j in range(x.shape[1])])
    rng = np.random.default_rng(np.random.SeedSequence([s, c]))
    u = rng.beta(r / f, (len(x) + 1 - r) / f)
    assigned = assign_columns(u, x, i)
    sorted_x = np.sort(x, axis=0)
    slots = np.minimum(np.floor(len(x) * u).astype(np.int64), len(x) - 1)
    inverse = sorted_x[slots, np.arange(x.shape[1])].astype(np.int64)
    return {'latent': u, 'assigned': assigned, 'inverse': inverse}


def _distance_inputs(reference, scale):
    r = _finite_array(reference, 'reference', 2)
    s = _finite_array(scale, 'scale', 1)
    if min(r.shape) < 1 or s.shape != (r.shape[1],) or np.any(s <= 0):
        raise ValueError('Nonempty reference and exact positive scale required')
    z = r / s
    if not np.isfinite(z).all():
        raise ValueError('Scaled reference overflow')
    return r, s, z


def nearest_distances(reference, query, scale) -> np.ndarray:
    r, s, z = _distance_inputs(reference, scale)
    q = _finite_array(query, 'query', 2)
    if q.shape[0] < 1 or q.shape[1] != r.shape[1]:
        raise ValueError('Nonempty aligned query required')
    qz = q / s
    if not np.isfinite(qz).all():
        raise ValueError('Scaled query overflow')
    return cKDTree(z).query(qz, k=1, workers=1)[0]


def leave_one_out(reference, scale) -> np.ndarray:
    r, s, z = _distance_inputs(reference, scale)
    if len(r) == 1:
        return np.zeros(1)
    # The second distance is the nearest distinct row identity, including ties.
    return cKDTree(z).query(z, k=2, workers=1)[0][:, 1]


def beta_variance_ratio(n, width) -> float:
    n = _integer(n, 'n', 1)
    f = _real(width, 'width')
    if not 1 <= f <= n:
        raise ValueError('Bandwidth must be within [1,n]')
    return f * (n + 2) / (n + 1 + f)


def atomic_json(path, value) -> None:
    """Publish complete JSON without replacing existing evidence."""
    p = Path(path)
    if os.path.lexists(p):
        raise FileExistsError(str(p))
    # Validate serialization before creating even a temporary file.
    try:
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Value cannot be serialized as finite JSON') from exc
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.h004a_', dir=p.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is exclusive at the destination, unlike os.replace.
        os.link(tmp, p)
    finally:
        os.unlink(tmp)
