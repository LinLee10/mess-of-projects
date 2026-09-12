"""Auditable full coverage minibatches for the pinned published generators.

This does not change objectives, architectures or sampling equations. It replaces
within-class sampling with shuffled cycles for CTGAN, and row sampling with
shuffled cycles for TVAE/DDPM. All gradient-exposed row counts are retained.
Original source is hash checked; the derived executable source is exported.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

PINNED_BLOB = '9be7fd0245d378a0b6295edfb95b70e293ffb9b6'
_CACHE = {}

class CoverageCycle:
    def __init__(self,n,seed):
        if type(n) is not int or n<1:raise ValueError('Positive row count required')
        self.n=n;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(n)
        self.position=0;self.counts=np.zeros(n,dtype=np.int64)
    def take(self,n):
        if type(n) is not int or n<1:raise ValueError('Positive batch count required')
        pieces=[];remaining=n
        while remaining:
            if self.position==self.n:
                self.order=self.rng.permutation(self.n);self.position=0
            size=min(remaining,self.n-self.position)
            pieces.append(self.order[self.position:self.position+size]);self.position+=size;remaining-=size
        indices=np.concatenate(pieces);np.add.at(self.counts,indices,1);return indices
    def record(self,indices):np.add.at(self.counts,indices,1)

def cached_transform(x,y,config,seed):
    from synthlab_next.common import array_hash
    from synthlab_next.transforms import ModeTransformer
    key=(array_hash(x),array_hash(y),config.max_modes,seed)
    if key not in _CACHE:_CACHE[key]=ModeTransformer(config.max_modes,seed).fit(x,y).to_dict()
    return ModeTransformer.from_dict(copy.deepcopy(_CACHE[key]))

def replace_exact(source,old,new,count=1):
    if source.count(old)!=count:raise ValueError('Pinned source contract changed: '+old[:80])
    return source.replace(old,new)

def load_coverage_generators(export=None):
    import synthlab_next.generators as original
    path=Path(original.__file__);raw=path.read_bytes()
    sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if sha!=PINNED_BLOB:raise ValueError('Generator source is not the reviewed pinned implementation')
    source=raw.decode().replace('from .common import','from synthlab_next.common import').replace('from .transforms import','from synthlab_next.transforms import')
    source=source.replace('from __future__ import annotations','from __future__ import annotations\nfrom coverage_v2 import CoverageCycle, cached_transform')
    source=replace_exact(source,'self.rng=np.random.default_rng(self.seed)','self.rng=np.random.default_rng(self.seed)\n        self.coverage=CoverageCycle(len(self.y),self.seed+9701)')
    source=replace_exact(source,"        return self\n\n    def _check_n", "        if self.coverage.counts.min()<1:raise RuntimeError('Not all training rows received a gradient exposure')\n        self.audit.update(unique_training_rows_seen=int((self.coverage.counts>0).sum()),minimum_row_exposures=int(self.coverage.counts.min()),maximum_row_exposures=int(self.coverage.counts.max()),total_row_exposures=int(self.coverage.counts.sum()),coverage_counts=self.coverage.counts.tolist(),batch_policy='Full shuffled cycles; CTGAN retains log frequency class selection')\n        return self\n\n    def _check_n")
    source=replace_exact(source,'self.transformer=ModeTransformer(cfg.max_modes,self.seed).fit(self.x,self.y)','self.transformer=cached_transform(self.x,self.y,cfg,self.seed)',count=2)
    source=replace_exact(source,'            class_rows=[np.flatnonzero(self.y==c) for c in range(self.n_classes)]','            class_rows=[np.flatnonzero(self.y==c) for c in range(self.n_classes)]\n            cycles=[CoverageCycle(len(rows),self.seed+991+c) for c,rows in enumerate(class_rows)]')
    source=replace_exact(source,'                indices=np.array([self.rng.choice(class_rows[c]) for c in labels])','                indices=np.array([class_rows[c][cycles[c].take(1)[0]] for c in labels])\n                self.coverage.record(indices)')
    source=replace_exact(source,'                batch=data[torch.randint(len(data),(cfg.batch_size,))]','                batch=data[torch.tensor(self.coverage.take(cfg.batch_size))]')
    source=replace_exact(source,'                indices=torch.randint(len(data),(cfg.batch_size,));t=torch.randint(cfg.diffusion_steps,(cfg.batch_size,))','                indices=torch.tensor(self.coverage.take(cfg.batch_size));t=torch.randint(cfg.diffusion_steps,(cfg.batch_size,))')
    if export:
        p=Path(export);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
    name='coverage_generators_runtime'
    module=types.ModuleType(name);module.__file__=str(export or path);sys.modules[name]=module
    exec(compile(source,str(export or path),'exec'),module.__dict__)
    return module
