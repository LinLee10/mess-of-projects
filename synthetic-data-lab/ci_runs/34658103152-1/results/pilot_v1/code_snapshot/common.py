"""Strict inputs, deterministic provenance and immutable acceptance manifests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch


def validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x=np.asarray(x); y=np.asarray(y)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or len(x)<2 or x.shape[1]<1:
        raise ValueError('Expected a nonempty feature matrix and aligned one dimensional labels')
    if (not np.issubdtype(x.dtype,np.number) or not np.issubdtype(y.dtype,np.number)
            or np.iscomplexobj(x) or np.iscomplexobj(y)):
        raise ValueError('Real numeric arrays required; complex values cannot be discarded')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite data are forbidden')
    if np.issubdtype(y.dtype,np.bool_) or not np.equal(y,np.floor(y)).all():
        raise ValueError('Labels must be integers, not booleans or fractional values')
    classes=np.unique(y)
    if len(classes)<2 or not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError('Training labels must contain every integer class from zero')
    return np.array(x,dtype=np.float64,copy=True),np.array(y,dtype=np.int64,copy=True)


def validate_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    y=np.asarray(y)
    if y.ndim!=1 or len(y)==0 or not np.issubdtype(y.dtype,np.integer):
        raise ValueError('Requested labels must be a nonempty integer vector')
    if np.any(y<0) or np.any(y>=n_classes): raise ValueError('Unknown class requested')
    return y.astype(np.int64,copy=True)


def array_hash(x: np.ndarray) -> str:
    x=np.ascontiguousarray(x)
    h=hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(json.dumps(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def state_hash(module_or_state: Any) -> str:
    state=module_or_state.state_dict() if hasattr(module_or_state,'state_dict') else module_or_state
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def file_hash(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path: str|Path,value: Any) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def _safe_file(root: Path,name: str) -> Path:
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('Manifest path is missing, symbolic, or outside its run')
    return p


def seal(root: Path,paths: list[str]) -> dict:
    root=Path(root)
    if (root/'FROZEN.json').exists(): raise ValueError('Experiment already frozen')
    manifest={'schema_version':1,'files':{p:file_hash(_safe_file(root,p)) for p in sorted(set(paths))},
              'meaning':'Candidate and development decisions frozen before final test evaluation'}
    manifest['manifest_sha256']=canonical_hash(manifest)
    write_json(root/'FROZEN.json',manifest)
    return manifest


def verify_seal(root: Path) -> dict:
    root=Path(root)
    if not (root/'FROZEN.json').is_file(): raise ValueError('Final evaluation requires a frozen candidate manifest')
    manifest=json.loads((root/'FROZEN.json').read_text()); expected=manifest.pop('manifest_sha256',None)
    if canonical_hash(manifest)!=expected: raise ValueError('Frozen manifest was modified')
    for name,digest in manifest['files'].items():
        if file_hash(_safe_file(root,name))!=digest: raise ValueError(f'Frozen artifact changed: {name}')
    return dict(manifest,manifest_sha256=expected)


def optimize(loss: torch.Tensor,optimizer: torch.optim.Optimizer,parameters,clip: float=10.0) -> float:
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    parameters=list(parameters)
    norm=torch.nn.utils.clip_grad_norm_(parameters,clip)
    if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters): raise FloatingPointError('Nonfinite parameters')
    return float(loss.detach())
